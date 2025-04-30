/* Test basic nss_dns functionality with multiple threads.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* Unlike tst-resolv-basic, this test does not overwrite the _res
   structure and relies on namespaces to achieve the redirection to
   the test servers with a custom /etc/resolv.conf file.  */

#include <dlfcn.h>
#include <errno.h>
#include <gnu/lib-names.h>
#include <netdb.h>
#include <resolv/resolv-internal.h>
#include <resolv/resolv_context.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/resolv_test.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xthread.h>
#include <support/xunistd.h>

/* Each client thread sends this many queries.  */
enum { queries_per_thread = 500 };

/* Return a small positive number identifying this thread.  */
static int
get_thread_number (void)
{
  static int __thread local;
  if (local != 0)
    return local;
  static int global = 1;
  local = __atomic_fetch_add (&global, 1, __ATOMIC_RELAXED);
  return local;
}

static void
response (const struct resolv_response_context *ctx,
          struct resolv_response_builder *b,
          const char *qname, uint16_t qclass, uint16_t qtype)
{
  TEST_VERIFY_EXIT (qname != NULL);

  int counter = 0;
  int thread = 0;
  int dummy = 0;
  TEST_VERIFY (sscanf (qname, "counter%d.thread%d.example.com%n",
                       &counter, &thread, &dummy) == 2);
  TEST_VERIFY (dummy > 0);

  struct resolv_response_flags flags = { 0 };
  resolv_response_init (b, flags);
  resolv_response_add_question (b, qname, qclass, qtype);

  resolv_response_section (b, ns_s_an);
  resolv_response_open_record (b, qname, qclass, qtype, 0);
  switch (qtype)
    {
    case T_A:
      {
        char ipv4[4] = {10, 0, counter, thread};
        resolv_response_add_data (b, &ipv4, sizeof (ipv4));
      }
      break;
    case T_AAAA:
      {
        char ipv6[16]
          = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0, 0,
             counter, 0, thread, 0, 0};
        resolv_response_add_data (b, &ipv6, sizeof (ipv6));
      }
      break;
    default:
      support_record_failure ();
      printf ("error: unexpected QTYPE: %s/%u/%u\n",
              qname, qclass, qtype);
    }
  resolv_response_close_record (b);
}

/* Check that the resolver configuration for this thread has an
   extended resolver configuration.  */
static void
check_have_conf (void)
{
  struct resolv_context *ctx = __resolv_context_get ();
  TEST_VERIFY_EXIT (ctx != NULL);
  TEST_VERIFY (ctx->conf != NULL);
  __resolv_context_put (ctx);
}

/* Verify that E matches the expected response for FAMILY and
   COUNTER.  */
static void
check_hostent (const char *caller, const char *function, const char *qname,
               int ret, struct hostent *e, int family, int counter)
{
  if (ret != 0)
    {
      errno = ret;
      support_record_failure ();
      printf ("error: %s: %s for %s failed: %m\n", caller, function, qname);
      return;
    }

  TEST_VERIFY_EXIT (e != NULL);
  TEST_VERIFY (strcmp (qname, e->h_name) == 0);
  TEST_VERIFY (e->h_addrtype == family);
  TEST_VERIFY_EXIT (e->h_addr_list[0] != NULL);
  TEST_VERIFY (e->h_addr_list[1] == NULL);
  switch (family)
    {
    case AF_INET:
      {
        char addr[4] = {10, 0, counter, get_thread_number ()};
        TEST_VERIFY (e->h_length == sizeof (addr));
        TEST_VERIFY (memcmp (e->h_addr_list[0], addr, sizeof (addr)) == 0);
      }
      break;
    case AF_INET6:
      {
      char addr[16]
        = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0,
           0, counter, 0, get_thread_number (), 0, 0};
      TEST_VERIFY (e->h_length == sizeof (addr));
      TEST_VERIFY (memcmp (e->h_addr_list[0], addr, sizeof (addr)) == 0);
      }
      break;
    default:
      FAIL_EXIT1 ("%s: invalid address family %d", caller, family);
    }
  check_have_conf ();
}

/* Check a getaddrinfo result.  */
static void
check_addrinfo (const char *caller, const char *qname,
                int ret, struct addrinfo *ai, int family, int counter)
{
  if (ret != 0)
    {
      support_record_failure ();
      printf ("error: %s: getaddrinfo for %s failed: %s\n",
              caller, qname, gai_strerror (ret));
      return;
    }

  TEST_VERIFY_EXIT (ai != NULL);

  /* Check that available data matches the requirements.  */
  bool have_ipv4 = false;
  bool have_ipv6 = false;
  for (struct addrinfo *p = ai; p != NULL; p = p->ai_next)
    {
      TEST_VERIFY (p->ai_socktype == SOCK_STREAM);
      TEST_VERIFY (p->ai_protocol == IPPROTO_TCP);
      TEST_VERIFY_EXIT (p->ai_addr != NULL);
      TEST_VERIFY (p->ai_addr->sa_family == p->ai_family);

      switch (p->ai_family)
        {
        case AF_INET:
          {
            TEST_VERIFY (!have_ipv4);
            have_ipv4 = true;
            struct sockaddr_in *sa = (struct sockaddr_in *) p->ai_addr;
            TEST_VERIFY (p->ai_addrlen == sizeof (*sa));
            char addr[4] = {10, 0, counter, get_thread_number ()};
            TEST_VERIFY (memcmp (&sa->sin_addr, addr, sizeof (addr)) == 0);
            TEST_VERIFY (ntohs (sa->sin_port) == 80);
          }
          break;
        case AF_INET6:
          {
            TEST_VERIFY (!have_ipv6);
            have_ipv6 = true;
            struct sockaddr_in6 *sa = (struct sockaddr_in6 *) p->ai_addr;
            TEST_VERIFY (p->ai_addrlen == sizeof (*sa));
            char addr[16]
              = {0x20, 0x01, 0xd, 0xb8, 0, 0, 0, 0, 0, 0,
                 0, counter, 0, get_thread_number (), 0, 0};
            TEST_VERIFY (memcmp (&sa->sin6_addr, addr, sizeof (addr)) == 0);
            TEST_VERIFY (ntohs (sa->sin6_port) == 80);
          }
          break;
        default:
          FAIL_EXIT1 ("%s: invalid address family %d", caller, family);
        }
    }

  switch (family)
    {
      case AF_INET:
        TEST_VERIFY (have_ipv4);
        TEST_VERIFY (!have_ipv6);
        break;
      case AF_INET6:
        TEST_VERIFY (!have_ipv4);
        TEST_VERIFY (have_ipv6);
        break;
      case AF_UNSPEC:
        TEST_VERIFY (have_ipv4);
        TEST_VERIFY (have_ipv6);
        break;
    default:
      FAIL_EXIT1 ("%s: invalid address family %d", caller, family);
    }

  check_have_conf ();
}

/* This barrier ensures that all test threads begin their work
   simultaneously.  */
static pthread_barrier_t barrier;

/* Test gethostbyname2_r (if do_2 is false) or gethostbyname2_r with
   AF_INET (if do_2 is true).  */
static void *
byname (bool do_2)
{
  int this_thread = get_thread_number ();
  xpthread_barrier_wait (&barrier);
  for (int i = 0; i < queries_per_thread; ++i)
    {
      char qname[100];
      snprintf (qname, sizeof (qname), "counter%d.thread%d.example.com",
                i, this_thread);
      struct hostent storage;
      char buf[1000];
      struct hostent *e = NULL;
      int herrno;
      int ret;
      if (do_2)
        ret = gethostbyname_r (qname, &storage, buf, sizeof (buf),
                               &e, &herrno);
      else
        ret = gethostbyname2_r (qname, AF_INET, &storage, buf, sizeof (buf),
                                &e, &herrno);
      check_hostent (__func__, do_2 ? "gethostbyname2_r" : "gethostbyname_r",
                     qname, ret, e, AF_INET, i);
    }
  check_have_conf ();
  return NULL;
}

/* Test gethostbyname_r.  */
static void *
thread_byname (void *closure)
{
  return byname (false);
}

/* Test gethostbyname2_r with AF_INET.  */
static void *
thread_byname2 (void *closure)
{
  return byname (true);
}

/* Test gethostbyname2_r with AF_INET6.  */
static void *
thread_byname2_af_inet6 (void *closure)
{
  int this_thread = get_thread_number ();
  xpthread_barrier_wait (&barrier);
  for (int i = 0; i < queries_per_thread; ++i)
    {
      char qname[100];
      snprintf (qname, sizeof (qname), "counter%d.thread%d.example.com",
                i, this_thread);
      struct hostent storage;
      char buf[1000];
      struct hostent *e = NULL;
      int herrno;
      int ret = gethostbyname2_r (qname, AF_INET6, &storage, buf, sizeof (buf),
                                  &e, &herrno);
      check_hostent (__func__, "gethostbyname2_r", qname, ret, e, AF_INET6, i);
    }
  return NULL;
}

/* Run getaddrinfo tests for FAMILY.  */
static void *
gai (int family)
{
  int this_thread = get_thread_number ();
  xpthread_barrier_wait (&barrier);
  for (int i = 0; i < queries_per_thread; ++i)
    {
      char qname[100];
      snprintf (qname, sizeof (qname), "counter%d.thread%d.example.com",
                i, this_thread);
      struct addrinfo hints =
        {
          .ai_family = family,
          .ai_socktype = SOCK_STREAM,
          .ai_protocol = IPPROTO_TCP,
        };
      struct addrinfo *ai;
      int ret = getaddrinfo (qname, "80", &hints, &ai);
      check_addrinfo (__func__, qname, ret, ai, family, i);
      if (ret == 0)
        freeaddrinfo (ai);
    }
  return NULL;
}

/* Test getaddrinfo with AF_INET.  */
static void *
thread_gai_inet (void *closure)
{
  return gai (AF_INET);
}

/* Test getaddrinfo with AF_INET6.  */
static void *
thread_gai_inet6 (void *closure)
{
  return gai (AF_INET6);
}

/* Test getaddrinfo with AF_UNSPEC.  */
static void *
thread_gai_unspec (void *closure)
{
  return gai (AF_UNSPEC);
}

/* Description of the chroot environment used to run the tests.  */
static struct support_chroot *chroot_env;

/* Set up the chroot environment.  */
static void
prepare (int argc, char **argv)
{
  chroot_env = support_chroot_create
    ((struct support_chroot_configuration)
     {
       .resolv_conf =
         "search example.com\n"
         "nameserver 127.0.0.1\n"
         "nameserver 127.0.0.2\n"
         "nameserver 127.0.0.3\n",
     });
}

static int
do_test (void)
{
  support_become_root ();
  if (!support_enter_network_namespace ())
    return EXIT_UNSUPPORTED;
  if (!support_can_chroot ())
    return EXIT_UNSUPPORTED;

  /* Load the shared object outside of the chroot.  */
  TEST_VERIFY (dlopen (LIBNSS_DNS_SO, RTLD_LAZY) != NULL);

  xchroot (chroot_env->path_chroot);
  TEST_VERIFY_EXIT (chdir ("/") == 0);

  struct sockaddr_in server_address =
    {
      .sin_family = AF_INET,
      .sin_addr = { .s_addr = htonl (INADDR_LOOPBACK) },
      .sin_port = htons (53)
    };
  const struct sockaddr *server_addresses[1] =
    { (const struct sockaddr *) &server_address };

  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,
       .nscount = 1,
       .disable_redirect = true,
       .server_address_overrides = server_addresses,
     });

  enum { thread_count = 6 };
  xpthread_barrier_init (&barrier, NULL, thread_count + 1);
  pthread_t threads[thread_count];
  typedef void *(*thread_func) (void *);
  thread_func thread_funcs[thread_count] =
    {
      thread_byname,
      thread_byname2,
      thread_byname2_af_inet6,
      thread_gai_inet,
      thread_gai_inet6,
      thread_gai_unspec,
    };
  for (int i = 0; i < thread_count; ++i)
    threads[i] = xpthread_create (NULL, thread_funcs[i], NULL);
  xpthread_barrier_wait (&barrier); /* Start the test threads.  */
  for (int i = 0; i < thread_count; ++i)
    xpthread_join (threads[i]);

  resolv_test_end (aux);
  support_chroot_free (chroot_env);

  return 0;
}

#define PREPARE prepare
#include <support/test-driver.c>
