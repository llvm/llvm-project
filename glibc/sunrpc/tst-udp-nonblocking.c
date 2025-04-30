/* Test non-blocking use of the UDP client.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <netinet/in.h>
#include <rpc/clnt.h>
#include <rpc/svc.h>
#include <stdbool.h>
#include <string.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/test-driver.h>
#include <support/xsocket.h>
#include <support/xunistd.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

/* Test data serialization and deserialization.   */

struct test_query
{
  uint32_t a;
  uint32_t b;
  uint32_t timeout_ms;
};

static bool_t
xdr_test_query (XDR *xdrs, void *data, ...)
{
  struct test_query *p = data;
  return xdr_uint32_t (xdrs, &p->a)
    && xdr_uint32_t (xdrs, &p->b)
    && xdr_uint32_t (xdrs, &p->timeout_ms);
}

struct test_response
{
  uint32_t server_id;
  uint32_t seq;
  uint32_t sum;
};

static bool_t
xdr_test_response (XDR *xdrs, void *data, ...)
{
  struct test_response *p = data;
  return xdr_uint32_t (xdrs, &p->server_id)
    && xdr_uint32_t (xdrs, &p->seq)
    && xdr_uint32_t (xdrs, &p->sum);
}

/* Implementation of the test server.  */

enum
  {
    /* Number of test servers to run. */
    SERVER_COUNT = 3,

    /* RPC parameters, chosen at random.  */
    PROGNUM = 8242,
    VERSNUM = 19654,

    /* Main RPC operation.  */
    PROC_ADD = 1,

    /* Request process termination.  */
    PROC_EXIT,

    /* Special exit status to mark successful processing.  */
    EXIT_MARKER = 55,
  };

/* Set by the parent process to tell test servers apart.  */
static int server_id;

/* Implementation of the test server.  */
static void
server_dispatch (struct svc_req *request, SVCXPRT *transport)
{
  /* Query sequence number.  */
  static uint32_t seq = 0;
  ++seq;
  static bool proc_add_seen;

  if (test_verbose)
    printf ("info: server_dispatch server_id=%d seq=%u rq_proc=%lu\n",
            server_id, seq, request->rq_proc);

  switch (request->rq_proc)
    {
    case PROC_ADD:
      {
        struct test_query query;
        memset (&query, 0xc0, sizeof (query));
        TEST_VERIFY_EXIT
          (svc_getargs (transport, xdr_test_query,
                        (void *) &query));

        if (test_verbose)
          printf ("  a=%u b=%u timeout_ms=%u\n",
                  query.a, query.b, query.timeout_ms);

        usleep (query.timeout_ms * 1000);

        struct test_response response =
          {
            .server_id = server_id,
            .seq = seq,
            .sum = query.a + query.b,
          };
        TEST_VERIFY (svc_sendreply (transport, xdr_test_response,
                                    (void *) &response));
        if (test_verbose)
          printf ("  server id %d response seq=%u sent\n", server_id, seq);
        proc_add_seen = true;
      }
      break;

    case PROC_EXIT:
      TEST_VERIFY (proc_add_seen);
      TEST_VERIFY (svc_sendreply (transport, (xdrproc_t) xdr_void, NULL));
      _exit (EXIT_MARKER);
      break;

    default:
      FAIL_EXIT1 ("invalid rq_proc value: %lu", request->rq_proc);
      break;
    }
}

/* Return the number seconds since an arbitrary point in time.  */
static double
get_ticks (void)
{
  {
    struct timespec ts;
    if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
      return ts.tv_sec + ts.tv_nsec * 1e-9;
  }
  {
    struct timeval tv;
    TEST_VERIFY_EXIT (gettimeofday (&tv, NULL) == 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
  }
}

static int
do_test (void)
{
  support_become_root ();
  support_enter_network_namespace ();

  /* Information about the test servers.  */
  struct
  {
    SVCXPRT *transport;
    struct sockaddr_in address;
    pid_t pid;
    uint32_t xid;
  } servers[SERVER_COUNT];

  /* Spawn the test servers.  */
  for (int i = 0; i < SERVER_COUNT; ++i)
    {
      servers[i].transport = svcudp_create (RPC_ANYSOCK);
      TEST_VERIFY_EXIT (servers[i].transport != NULL);
      servers[i].address = (struct sockaddr_in)
        {
          .sin_family = AF_INET,
          .sin_addr.s_addr = htonl (INADDR_LOOPBACK),
          .sin_port = htons (servers[i].transport->xp_port),
        };
      servers[i].xid = 0xabcd0101 + i;
      if (test_verbose)
        printf ("info: setting up server %d xid=%x on port %d\n",
                i, servers[i].xid, servers[i].transport->xp_port);

      server_id = i;
      servers[i].pid = xfork ();
      if (servers[i].pid == 0)
        {
          TEST_VERIFY (svc_register (servers[i].transport,
                                     PROGNUM, VERSNUM, server_dispatch, 0));
          svc_run ();
          FAIL_EXIT1 ("supposed to be unreachable");
        }
      /* We need to close the socket so that we do not accidentally
         consume the request.  */
      TEST_VERIFY (close (servers[i].transport->xp_sock) == 0);
    }


  /* The following code mirrors what ypbind does.  */

  /* Copied from clnt_udp.c (like ypbind).  */
  struct cu_data
  {
    int cu_sock;
    bool_t cu_closeit;
    struct sockaddr_in cu_raddr;
    int cu_rlen;
    struct timeval cu_wait;
    struct timeval cu_total;
    struct rpc_err cu_error;
    XDR cu_outxdrs;
    u_int cu_xdrpos;
    u_int cu_sendsz;
    char *cu_outbuf;
    u_int cu_recvsz;
    char cu_inbuf[1];
  };

  int client_socket = xsocket (AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
  CLIENT *clnt = clntudp_create (&servers[0].address, PROGNUM, VERSNUM,
                                 /* 5 seconds per-response timeout.  */
                                 ((struct timeval) { 5, 0 }),
                                 &client_socket);
  TEST_VERIFY (clnt != NULL);
  clnt->cl_auth = authunix_create_default ();
  {
    struct timeval zero = { 0, 0 };
    TEST_VERIFY (clnt_control (clnt, CLSET_TIMEOUT, (void *) &zero));
  }

  /* Poke at internal data structures (like ypbind).  */
  struct cu_data *cu = (struct cu_data *) clnt->cl_private;

  /* Send a ping to each server.  */
  double before_pings = get_ticks ();
  for (int i = 0; i < SERVER_COUNT; ++i)
    {
      if (test_verbose)
        printf ("info: sending server %d ping\n", i);
      /* Reset the xid because it is changed by each invocation of
         clnt_call.  Subtract one to compensate for the xid update
         during the call.  */
      *((uint32_t *) (cu->cu_outbuf)) = servers[i].xid - 1;
      cu->cu_raddr = servers[i].address;

      struct test_query query = { .a = 100, .b = i + 1 };
      if (i == 1)
        /* Shorter timeout to prefer this server.  These timeouts must
           be much shorter than the 5-second per-response timeout
           configured with clntudp_create.  */
        query.timeout_ms = 750;
      else
        query.timeout_ms = 1500;
      struct test_response response = { 0 };
      /* NB: Do not check the return value.  The server reply will
         prove that the call worked.  */
      double before_one_ping = get_ticks ();
      clnt_call (clnt, PROC_ADD,
                 xdr_test_query, (void *) &query,
                 xdr_test_response, (void *) &response,
                 ((struct timeval) { 0, 0 }));
      double after_one_ping = get_ticks ();
      if (test_verbose)
        printf ("info: non-blocking send took %f seconds\n",
                after_one_ping - before_one_ping);
      /* clnt_call should return immediately.  Accept some delay in
         case the process is descheduled.  */
      TEST_VERIFY (after_one_ping - before_one_ping < 0.3);
    }

  /* Collect the non-blocking response.  */
  if (test_verbose)
    printf ("info: collecting response\n");
  struct test_response response = { 0 };
  TEST_VERIFY
    (clnt_call (clnt, PROC_ADD, NULL, NULL,
                xdr_test_response, (void *) &response,
                ((struct timeval) { 0, 0 })) == RPC_SUCCESS);
  double after_pings = get_ticks ();
  if (test_verbose)
    printf ("info: send/receive took %f seconds\n",
            after_pings - before_pings);
  /* Expected timeout is 0.75 seconds.  */
  TEST_VERIFY (0.70 <= after_pings - before_pings);
  TEST_VERIFY (after_pings - before_pings < 1.2);

  uint32_t xid;
  memcpy (&xid, &cu->cu_inbuf, sizeof (xid));
  if (test_verbose)
    printf ("info: non-blocking response: xid=%x server_id=%u seq=%u sum=%u\n",
            xid, response.server_id, response.seq, response.sum);
  /* Check that the reply from the preferred server was used.  */
  TEST_VERIFY (servers[1].xid == xid);
  TEST_VERIFY (response.server_id == 1);
  TEST_VERIFY (response.seq == 1);
  TEST_VERIFY (response.sum == 102);

  auth_destroy (clnt->cl_auth);
  clnt_destroy (clnt);

  for (int i = 0; i < SERVER_COUNT; ++i)
    {
      if (test_verbose)
        printf ("info: requesting server %d termination\n", i);
      client_socket = RPC_ANYSOCK;
      clnt = clntudp_create (&servers[i].address, PROGNUM, VERSNUM,
                             ((struct timeval) { 5, 0 }),
                             &client_socket);
      TEST_VERIFY_EXIT (clnt != NULL);
      TEST_VERIFY (clnt_call (clnt, PROC_EXIT,
                              (xdrproc_t) xdr_void, NULL,
                              (xdrproc_t) xdr_void, NULL,
                              ((struct timeval) { 3, 0 })) == RPC_SUCCESS);
      clnt_destroy (clnt);

      int status;
      xwaitpid (servers[i].pid, &status, 0);
      TEST_VERIFY (WIFEXITED (status) && WEXITSTATUS (status) == EXIT_MARKER);
    }

  return 0;
}

#include <support/test-driver.c>
