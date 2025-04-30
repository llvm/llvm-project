/* Test timeout handling in the UDP client.
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
#include <stdlib.h>

static pid_t server_pid;

/* Test data serialization and deserialization.   */

struct test_query
{
  uint32_t a;
  uint32_t b;
  uint32_t timeout_ms;
  uint32_t wait_for_seq;
  uint32_t garbage_packets;
};

static bool_t
xdr_test_query (XDR *xdrs, void *data, ...)
{
  struct test_query *p = data;
  return xdr_uint32_t (xdrs, &p->a)
    && xdr_uint32_t (xdrs, &p->b)
    && xdr_uint32_t (xdrs, &p->timeout_ms)
    && xdr_uint32_t (xdrs, &p->wait_for_seq)
    && xdr_uint32_t (xdrs, &p->garbage_packets);
}

struct test_response
{
  uint32_t seq;
  uint32_t sum;
};

static bool_t
xdr_test_response (XDR *xdrs, void *data, ...)
{
  struct test_response *p = data;
  return xdr_uint32_t (xdrs, &p->seq)
    && xdr_uint32_t (xdrs, &p->sum);
}

/* Implementation of the test server.  */

enum
  {
    /* RPC parameters, chosen at random.  */
    PROGNUM = 15717,
    VERSNUM = 13689,

    /* Main RPC operation.  */
    PROC_ADD = 1,

    /* Reset the sequence number.  */
    PROC_RESET_SEQ,

    /* Request process termination.  */
    PROC_EXIT,

    /* Special exit status to mark successful processing.  */
    EXIT_MARKER = 55,
  };

static void
server_dispatch (struct svc_req *request, SVCXPRT *transport)
{
  /* Query sequence number.  */
  static uint32_t seq = 0;
  ++seq;

  if (test_verbose)
    printf ("info: server_dispatch seq=%u rq_proc=%lu\n",
            seq, request->rq_proc);

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
          printf ("  a=%u b=%u timeout_ms=%u wait_for_seq=%u"
                  " garbage_packets=%u\n",
                  query.a, query.b, query.timeout_ms, query.wait_for_seq,
                  query.garbage_packets);

        if (seq < query.wait_for_seq)
          {
            /* No response at this point.  */
            if (test_verbose)
              printf ("  skipped response\n");
            break;
          }

        if (query.garbage_packets > 0)
          {
            int per_packet_timeout;
            if (query.timeout_ms > 0)
              per_packet_timeout
                = query.timeout_ms * 1000 / query.garbage_packets;
            else
              per_packet_timeout = 0;

            char buf[20];
            memset (&buf, 0xc0, sizeof (buf));
            for (int i = 0; i < query.garbage_packets; ++i)
              {
                /* 13 is relatively prime to 20 = sizeof (buf) + 1, so
                   the len variable will cover the entire interval
                   [0, 20] if query.garbage_packets is sufficiently
                   large.  */
                size_t len = (i * 13 + 1) % (sizeof (buf) + 1);
                TEST_VERIFY (sendto (transport->xp_sock,
                                     buf, len, MSG_NOSIGNAL,
                                     (struct sockaddr *) &transport->xp_raddr,
                                     transport->xp_addrlen) == len);
                if (per_packet_timeout > 0)
                  usleep (per_packet_timeout);
              }
          }
        else if (query.timeout_ms > 0)
          usleep (query.timeout_ms * 1000);

        struct test_response response =
          {
            .seq = seq,
            .sum = query.a + query.b,
          };
        TEST_VERIFY (svc_sendreply (transport, xdr_test_response,
                                    (void *) &response));
      }
      break;

    case PROC_RESET_SEQ:
      seq = 0;
      TEST_VERIFY (svc_sendreply (transport, (xdrproc_t) xdr_void, NULL));
      break;

    case PROC_EXIT:
      TEST_VERIFY (svc_sendreply (transport, (xdrproc_t) xdr_void, NULL));
      _exit (EXIT_MARKER);
      break;

    default:
      FAIL_EXIT1 ("invalid rq_proc value: %lu", request->rq_proc);
      break;
    }
}

/* Function to be called before exit to make sure the
   server process is properly killed.  */
static void
kill_server (void)
{
  kill (server_pid, SIGTERM);
}

/* Implementation of the test client.  */

static struct test_response
test_call (CLIENT *clnt, int proc, struct test_query query,
           struct timeval timeout)
{
  if (test_verbose)
    printf ("info: test_call proc=%d timeout=%lu.%06lu\n",
            proc, (unsigned long) timeout.tv_sec,
            (unsigned long) timeout.tv_usec);
  struct test_response response;
  TEST_VERIFY_EXIT (clnt_call (clnt, proc,
                               xdr_test_query, (void *) &query,
                               xdr_test_response, (void *) &response,
                               timeout)
                    == RPC_SUCCESS);
  return response;
}

static void
test_call_timeout (CLIENT *clnt, int proc, struct test_query query,
                   struct timeval timeout)
{
  struct test_response response;
  TEST_VERIFY (clnt_call (clnt, proc,
                          xdr_test_query, (void *) &query,
                          xdr_test_response, (void *) &response,
                          timeout)
               == RPC_TIMEDOUT);
}

/* Complete one regular RPC call to drain the server socket
   buffer.  Resets the sequence number.  */
static void
test_call_flush (CLIENT *clnt)
{
  /* This needs a longer timeout to flush out all pending requests.
     The choice of 5 seconds is larger than the per-response timeouts
     requested via the timeout_ms field.  */
  if (test_verbose)
    printf ("info: flushing pending queries\n");
  TEST_VERIFY_EXIT (clnt_call (clnt, PROC_RESET_SEQ,
                               (xdrproc_t) xdr_void, NULL,
                               (xdrproc_t) xdr_void, NULL,
                               ((struct timeval) { 5, 0 }))
                    == RPC_SUCCESS);
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

static void
test_udp_server (int port)
{
  struct sockaddr_in sin =
    {
      .sin_family = AF_INET,
      .sin_addr.s_addr = htonl (INADDR_LOOPBACK),
      .sin_port = htons (port)
    };
  int sock = RPC_ANYSOCK;

  /* The client uses a 1.5 second timeout for retries.  The timeouts
     are arbitrary, but chosen so that there is a substantial gap
     between them, but the total time spent waiting is not too
     large.  */
  CLIENT *clnt = clntudp_create (&sin, PROGNUM, VERSNUM,
                                 (struct timeval) { 1, 500 * 1000 },
                                 &sock);
  TEST_VERIFY_EXIT (clnt != NULL);

  /* Basic call/response test.  */
  struct test_response response = test_call
    (clnt, PROC_ADD,
     (struct test_query) { .a = 17, .b = 4 },
     (struct timeval) { 3, 0 });
  TEST_VERIFY (response.sum == 21);
  TEST_VERIFY (response.seq == 1);

  /* Check that garbage packets do not interfere with timeout
     processing.  */
  double before = get_ticks ();
  response = test_call
    (clnt, PROC_ADD,
     (struct test_query) {
       .a = 19, .b = 4, .timeout_ms = 500, .garbage_packets = 21,
     },
     (struct timeval) { 3, 0 });
  TEST_VERIFY (response.sum == 23);
  TEST_VERIFY (response.seq == 2);
  double after = get_ticks ();
  if (test_verbose)
    printf ("info: 21 garbage packets took %f seconds\n", after - before);
  /* Expected timeout is 0.5 seconds.  Add some slack for rounding errors and
     in case process scheduling delays processing the query or response, but
     do not accept a retry (which would happen at 1.5 seconds).  */
  TEST_VERIFY (0.45 <= after - before);
  TEST_VERIFY (after - before < 1.2);
  test_call_flush (clnt);

  /* Check that missing a response introduces a 1.5 second timeout, as
     requested when calling clntudp_create.  */
  before = get_ticks ();
  response = test_call
    (clnt, PROC_ADD,
     (struct test_query) { .a = 170, .b = 40, .wait_for_seq = 2 },
     (struct timeval) { 3, 0 });
  TEST_VERIFY (response.sum == 210);
  TEST_VERIFY (response.seq == 2);
  after = get_ticks ();
  if (test_verbose)
    printf ("info: skipping one response took %f seconds\n",
            after - before);
  /* Expected timeout is 1.5 seconds.  Do not accept a second retry
     (which would happen at 3 seconds).  */
  TEST_VERIFY (1.45 <= after - before);
  TEST_VERIFY (after - before < 2.9);
  test_call_flush (clnt);

  /* Check that the overall timeout wins against the per-query
     timeout.  */
  before = get_ticks ();
  test_call_timeout
    (clnt, PROC_ADD,
     (struct test_query) { .a = 170, .b = 41, .wait_for_seq = 2 },
     (struct timeval) { 0, 750 * 1000 });
  after = get_ticks ();
  if (test_verbose)
    printf ("info: 0.75 second timeout took %f seconds\n",
            after - before);
  TEST_VERIFY (0.70 <= after - before);
  TEST_VERIFY (after - before < 1.4);
  test_call_flush (clnt);

  for (int with_garbage = 0; with_garbage < 2; ++with_garbage)
    {
      /* Check that no response at all causes the client to bail out.  */
      before = get_ticks ();
      test_call_timeout
        (clnt, PROC_ADD,
         (struct test_query) {
           .a = 170, .b = 40, .timeout_ms = 1200,
           .garbage_packets = with_garbage * 21
         },
         (struct timeval) { 0, 750 * 1000 });
      after = get_ticks ();
      if (test_verbose)
        printf ("info: test_udp_server: 0.75 second timeout took %f seconds"
                " (garbage %d)\n",
                after - before, with_garbage);
      TEST_VERIFY (0.70 <= after - before);
      TEST_VERIFY (after - before < 1.4);
      test_call_flush (clnt);

      /* As above, but check the total timeout.  */
      before = get_ticks ();
      test_call_timeout
        (clnt, PROC_ADD,
         (struct test_query) {
           .a = 170, .b = 40, .timeout_ms = 3000,
           .garbage_packets = with_garbage * 30
         },
         (struct timeval) { 2, 500 * 1000 });
      after = get_ticks ();
      if (test_verbose)
        printf ("info: test_udp_server: 2.5 second timeout took %f seconds"
                " (garbage %d)\n",
                after - before, with_garbage);
      TEST_VERIFY (2.45 <= after - before);
      TEST_VERIFY (after - before < 3.0);
      test_call_flush (clnt);
    }

  TEST_VERIFY_EXIT (clnt_call (clnt, PROC_EXIT,
                               (xdrproc_t) xdr_void, NULL,
                               (xdrproc_t) xdr_void, NULL,
                               ((struct timeval) { 5, 0 }))
                    == RPC_SUCCESS);
  clnt_destroy (clnt);
}

static int
do_test (void)
{
  support_become_root ();
  support_enter_network_namespace ();

  SVCXPRT *transport = svcudp_create (RPC_ANYSOCK);
  TEST_VERIFY_EXIT (transport != NULL);
  TEST_VERIFY (svc_register (transport, PROGNUM, VERSNUM, server_dispatch, 0));

  server_pid = xfork ();
  if (server_pid == 0)
    {
      svc_run ();
      FAIL_EXIT1 ("supposed to be unreachable");
    }
  atexit (kill_server);
  test_udp_server (transport->xp_port);

  int status;
  xwaitpid (server_pid, &status, 0);
  TEST_VERIFY (WIFEXITED (status) && WEXITSTATUS (status) == EXIT_MARKER);

  SVC_DESTROY (transport);
  return 0;
}

/* The minimum run time is around 17 seconds.  */
#define TIMEOUT 25
#include <support/test-driver.c>
