/* Test svc_register/svc_unregister rpcbind interaction (bug 5010).
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

/* This test uses a stub rpcbind server (implemented in a child
   process using rpcbind_dispatch/run_rpcbind) to check how RPC
   services are registered and unregistered using the rpcbind
   protocol.  For each subtest, a separate rpcbind test server is
   spawned and terminated.  */

#include <errno.h>
#include <netinet/in.h>
#include <rpc/clnt.h>
#include <rpc/pmap_prot.h>
#include <rpc/svc.h>
#include <signal.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/test-driver.h>
#include <support/xsocket.h>
#include <support/xthread.h>
#include <support/xunistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <libc-symbols.h>
#include <shlib-compat.h>

/* These functions are only available as compat symbols.  */
compat_symbol_reference (libc, xdr_pmap, xdr_pmap, GLIBC_2_0);
compat_symbol_reference (libc, svc_unregister, svc_unregister, GLIBC_2_0);

/* Server callback for the unused RPC service which is registered and
   unregistered.  */
static void
server_dispatch (struct svc_req *request, SVCXPRT *transport)
{
  FAIL_EXIT1 ("server_dispatch called");
}

/* The port on which rpcbind listens for incoming requests.  */
static inline struct sockaddr_in
rpcbind_address (void)
{
  return (struct sockaddr_in)
    {
      .sin_family = AF_INET,
      .sin_addr.s_addr = htonl (INADDR_LOOPBACK),
      .sin_port = htons (PMAPPORT)
    };
}

/* Data provided by the test server after running the test, to see
   that the expected calls (and only those) happened.  */
struct test_state
{
  bool_t set_called;
  bool_t unset_called;
};

static bool_t
xdr_test_state (XDR *xdrs, void *data, ...)
{
  struct test_state *p = data;
  return xdr_bool (xdrs, &p->set_called)
    && xdr_bool (xdrs, &p->unset_called);
}

enum
{
  /* Coordinates of our test service.  These numbers are
     arbitrary.  */
  PROGNUM = 123,
  VERSNUM = 456,

  /* Extension for this test.  */
  PROC_GET_STATE_AND_EXIT = 10760
};

/* Dummy implementation of the rpcbind service, with the
   PROC_GET_STATE_AND_EXIT extension.  */
static void
rpcbind_dispatch (struct svc_req *request, SVCXPRT *transport)
{
  static struct test_state state = { 0, };

  if (test_verbose)
    printf ("info: rpcbind request %lu\n", request->rq_proc);

  switch (request->rq_proc)
    {
    case PMAPPROC_SET:
    case PMAPPROC_UNSET:
      TEST_VERIFY (state.set_called == (request->rq_proc == PMAPPROC_UNSET));
      TEST_VERIFY (!state.unset_called);

      struct pmap query = { 0, };
      TEST_VERIFY
        (svc_getargs (transport, (xdrproc_t) xdr_pmap, (void *) &query));
      if (test_verbose)
        printf ("  pm_prog=%lu pm_vers=%lu pm_prot=%lu pm_port=%lu\n",
                query.pm_prog, query.pm_vers, query.pm_prot, query.pm_port);
      TEST_VERIFY (query.pm_prog == PROGNUM);
      TEST_VERIFY (query.pm_vers == VERSNUM);

      if (request->rq_proc == PMAPPROC_SET)
        state.set_called = TRUE;
      else
        state.unset_called = TRUE;

      bool_t result = TRUE;
      TEST_VERIFY (svc_sendreply (transport,
                                  (xdrproc_t) xdr_bool, (void *) &result));
      break;

    case PROC_GET_STATE_AND_EXIT:
      TEST_VERIFY (svc_sendreply (transport,
                                  xdr_test_state, (void *) &state));
      _exit (0);
      break;

    default:
      FAIL_EXIT1 ("invalid rq_proc value: %lu", request->rq_proc);
    }
}

/* Run the rpcbind test server.  */
static void
run_rpcbind (int rpcbind_sock)
{
  SVCXPRT *rpcbind_transport = svcudp_create (rpcbind_sock);
  TEST_VERIFY (svc_register (rpcbind_transport, PMAPPROG, PMAPVERS,
                             rpcbind_dispatch,
                             /* Do not register with rpcbind.  */
                             0));
  svc_run ();
}

/* Call out to the rpcbind test server to retrieve the test status
   information.  */
static struct test_state
get_test_state (void)
{
  int socket = RPC_ANYSOCK;
  struct sockaddr_in address = rpcbind_address ();
  CLIENT *client = clntudp_create
    (&address, PMAPPROG, PMAPVERS, (struct timeval) { 1, 0}, &socket);
  struct test_state result = { 0 };
  TEST_VERIFY (clnt_call (client, PROC_GET_STATE_AND_EXIT,
                          (xdrproc_t) xdr_void, NULL,
                          xdr_test_state, (void *) &result,
                          ((struct timeval) { 3, 0}))
               == RPC_SUCCESS);
  clnt_destroy (client);
  return result;
}

/* Used by test_server_thread to receive test parameters.  */
struct test_server_args
{
  bool use_rpcbind;
  bool use_unregister;
};

/* RPC test server.  Used to verify the svc_unregister behavior during
   thread cleanup.  */
static void *
test_server_thread (void *closure)
{
  struct test_server_args *args = closure;
  SVCXPRT *transport = svcudp_create (RPC_ANYSOCK);
  int protocol;
  if (args->use_rpcbind)
    protocol = IPPROTO_UDP;
  else
    /* Do not register with rpcbind.  */
    protocol = 0;
  TEST_VERIFY (svc_register (transport, PROGNUM, VERSNUM,
                             server_dispatch, protocol));
  if (args->use_unregister)
    svc_unregister (PROGNUM, VERSNUM);
  SVC_DESTROY (transport);
  return NULL;
}

static int
do_test (void)
{
  support_become_root ();
  support_enter_network_namespace ();

  /* Try to bind to the rpcbind port.  */
  int rpcbind_sock = xsocket (AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  {
    struct sockaddr_in sin = rpcbind_address ();
    if (bind (rpcbind_sock, (struct sockaddr *) &sin, sizeof (sin)) != 0)
      {
        /* If the port is not available, we cannot run this test.  */
        printf ("warning: could not bind to rpcbind port %d: %m\n",
                (int) PMAPPORT);
        return EXIT_UNSUPPORTED;
      }
  }

  for (int use_thread = 0; use_thread < 2; ++use_thread)
    for (int use_rpcbind = 0; use_rpcbind < 2; ++use_rpcbind)
      for (int use_unregister = 0; use_unregister < 2; ++use_unregister)
        {
          if (test_verbose)
            printf ("info: * use_thread=%d use_rpcbind=%d use_unregister=%d\n",
                    use_thread, use_rpcbind, use_unregister);

          /* Create the subprocess which runs the actual test.  The
             kernel will queue the UDP packets to the rpcbind
             process.  */
          pid_t svc_pid = xfork ();
          if (svc_pid == 0)
            {
              struct test_server_args args =
                {
                  .use_rpcbind = use_rpcbind,
                  .use_unregister = use_unregister,
                };
              if (use_thread)
                xpthread_join (xpthread_create
                               (NULL, test_server_thread, &args));
              else
                test_server_thread (&args);
              /* We cannnot use _exit here because we want to test the
                 process cleanup.  */
              exit (0);
            }

          /* Create the subprocess for the rpcbind test server.  */
          pid_t rpcbind_pid = xfork ();
          if (rpcbind_pid == 0)
            run_rpcbind (rpcbind_sock);

          int status;
          xwaitpid (svc_pid, &status, 0);
          TEST_VERIFY (WIFEXITED (status) && WEXITSTATUS (status) == 0);

          if (!use_rpcbind)
            /* Wait a bit, to see if the packet arrives on the rpcbind
               port.  The choice is of the timeout is arbitrary, but
               should be long enough even for slow/busy systems.  For
               the use_rpcbind case, waiting on svc_pid above makes
               sure that the test server has responded because
               svc_register/svc_unregister are supposed to wait for a
               reply.  */
            usleep (300 * 1000);

          struct test_state state = get_test_state ();
          if (use_rpcbind)
            {
              TEST_VERIFY (state.set_called);
              if (use_thread || use_unregister)
                /* Thread cleanup or explicit svc_unregister will
                   result in a rpcbind unset RPC call.  */
                TEST_VERIFY (state.unset_called);
              else
                /* This is arguably a bug: Regular process termination
                   does not unregister the service with rpcbind.  The
                   unset rpcbind call happens from a __libc_subfreeres
                   callback, and this only happens when running under
                   memory debuggers such as valgrind.  */
                TEST_VERIFY (!state.unset_called);
            }
          else
            {
              /* If rpcbind registration is not requested, we do not
                 expect any rpcbind calls.  */
              TEST_VERIFY (!state.set_called);
              TEST_VERIFY (!state.unset_called);
            }

          xwaitpid (rpcbind_pid, &status, 0);
          TEST_VERIFY (WIFEXITED (status) && WEXITSTATUS (status) == 0);
        }

  return 0;
}

#include <support/test-driver.c>
