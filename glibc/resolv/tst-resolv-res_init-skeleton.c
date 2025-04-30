/* Test parsing of /etc/resolv.conf.  Genric version.
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

/* Before including this file, TEST_THREAD has to be defined to 0 or
   1, depending on whether the threading tests should be compiled
   in.  */

#include <arpa/inet.h>
#include <errno.h>
#include <gnu/lib-names.h>
#include <netdb.h>
#include <resolv/resolv_context.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/run_diff.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xsocket.h>
#include <support/xstdio.h>
#include <support/xunistd.h>

#if TEST_THREAD
# include <support/xthread.h>
#endif

/* This is the host name used to ensure predictable behavior of
   res_init.  */
static const char *const test_hostname = "www.example.com";

struct support_chroot *chroot_env;

static void
prepare (int argc, char **argv)
{
  chroot_env = support_chroot_create
    ((struct support_chroot_configuration)
     {
       .resolv_conf = "",
     });
}

/* Verify that the chroot environment has been set up.  */
static void
check_chroot_working (void *closure)
{
  xchroot (chroot_env->path_chroot);
  FILE *fp = xfopen (_PATH_RESCONF, "r");
  xfclose (fp);

  TEST_VERIFY_EXIT (res_init () == 0);
  TEST_VERIFY (_res.options & RES_INIT);

  char buf[100];
  if (gethostname (buf, sizeof (buf)) < 0)
    FAIL_EXIT1 ("gethostname: %m");
  if (strcmp (buf, test_hostname) != 0)
    FAIL_EXIT1 ("unexpected host name: %s", buf);
}

/* If FLAG is set in *OPTIONS, write NAME to FP, and clear it in
   *OPTIONS.  */
static void
print_option_flag (FILE *fp, int *options, int flag, const char *name)
{
  if (*options & flag)
    {
      fprintf (fp, " %s", name);
      *options &= ~flag;
    }
}

/* Write a decoded version of the resolver configuration *RESP to the
   stream FP.  */
static void
print_resp (FILE *fp, res_state resp)
{
  struct resolv_context *ctx = __resolv_context_get_override (resp);
  TEST_VERIFY_EXIT (ctx != NULL);
  if (ctx->conf == NULL)
    fprintf (fp, "; extended resolver state missing\n");

  /* The options directive.  */
  {
    /* RES_INIT is used internally for tracking initialization.  */
    TEST_VERIFY (resp->options & RES_INIT);
    /* Also mask out other default flags which cannot be set through
       the options directive.  */
    int options
      = resp->options & ~(RES_INIT | RES_RECURSE | RES_DEFNAMES | RES_DNSRCH);
    if (options != 0
        || resp->ndots != 1
        || resp->retrans != RES_TIMEOUT
        || resp->retry != RES_DFLRETRY)
      {
        fputs ("options", fp);
        if (resp->ndots != 1)
          fprintf (fp, " ndots:%d", resp->ndots);
        if (resp->retrans != RES_TIMEOUT)
          fprintf (fp, " timeout:%d", resp->retrans);
        if (resp->retry != RES_DFLRETRY)
          fprintf (fp, " attempts:%d", resp->retry);
        print_option_flag (fp, &options, RES_USEVC, "use-vc");
        print_option_flag (fp, &options, RES_ROTATE, "rotate");
        print_option_flag (fp, &options, RES_USE_EDNS0, "edns0");
        print_option_flag (fp, &options, RES_SNGLKUP,
                           "single-request");
        print_option_flag (fp, &options, RES_SNGLKUPREOP,
                           "single-request-reopen");
        print_option_flag (fp, &options, RES_NOTLDQUERY, "no-tld-query");
        print_option_flag (fp, &options, RES_NORELOAD, "no-reload");
        print_option_flag (fp, &options, RES_TRUSTAD, "trust-ad");
        fputc ('\n', fp);
        if (options != 0)
          fprintf (fp, "; error: unresolved option bits: 0x%x\n", options);
      }
  }

  /* The search and domain directives.  */
  if (resp->dnsrch[0] != NULL)
    {
      fputs ("search", fp);
      for (int i = 0; i < MAXDNSRCH && resp->dnsrch[i] != NULL; ++i)
        {
          fputc (' ', fp);
          fputs (resp->dnsrch[i], fp);
        }
      fputc ('\n', fp);
    }
  else if (resp->defdname[0] != '\0')
    fprintf (fp, "domain %s\n", resp->defdname);

  /* The extended search path.  */
  {
    size_t i = 0;
    while (true)
      {
        const char *name = __resolv_context_search_list (ctx, i);
        if (name == NULL)
          break;
        fprintf (fp, "; search[%zu]: %s\n", i, name);
        ++i;
      }
  }

  /* The sortlist directive.  */
  if (resp->nsort > 0)
    {
      fputs ("sortlist", fp);
      for (int i = 0; i < resp->nsort && i < MAXRESOLVSORT; ++i)
        {
          char net[20];
          if (inet_ntop (AF_INET, &resp->sort_list[i].addr,
                         net, sizeof (net)) == NULL)
            FAIL_EXIT1 ("inet_ntop: %m\n");
          char mask[20];
          if (inet_ntop (AF_INET, &resp->sort_list[i].mask,
                         mask, sizeof (mask)) == NULL)
            FAIL_EXIT1 ("inet_ntop: %m\n");
          fprintf (fp, " %s/%s", net, mask);
        }
      fputc ('\n', fp);
    }

  /* The nameserver directives.  */
  for (size_t i = 0; i < resp->nscount; ++i)
    {
      char host[NI_MAXHOST];
      char service[NI_MAXSERV];

      /* See get_nsaddr in res_send.c.  */
      void *addr;
      size_t addrlen;
      if (resp->nsaddr_list[i].sin_family == 0
          && resp->_u._ext.nsaddrs[i] != NULL)
        {
          addr = resp->_u._ext.nsaddrs[i];
          addrlen = sizeof (*resp->_u._ext.nsaddrs[i]);
        }
      else
        {
          addr = &resp->nsaddr_list[i];
          addrlen = sizeof (resp->nsaddr_list[i]);
        }

      int ret = getnameinfo (addr, addrlen,
                             host, sizeof (host), service, sizeof (service),
                             NI_NUMERICHOST | NI_NUMERICSERV);
      if (ret != 0)
        {
          if (ret == EAI_SYSTEM)
            fprintf (fp, "; error: getnameinfo: %m\n");
          else
            fprintf (fp, "; error: getnameinfo: %s\n", gai_strerror (ret));
        }
      else
        {
          fprintf (fp, "nameserver %s\n", host);
          if (strcmp (service, "53") != 0)
            fprintf (fp, "; unrepresentable port number %s\n\n", service);
        }
    }

  /* The extended name server list.  */
  {
    size_t i = 0;
    while (true)
      {
        const struct sockaddr *addr = __resolv_context_nameserver (ctx, i);
        if (addr == NULL)
          break;
        size_t addrlen;
        switch (addr->sa_family)
          {
          case AF_INET:
            addrlen = sizeof (struct sockaddr_in);
            break;
          case AF_INET6:
            addrlen = sizeof (struct sockaddr_in6);
            break;
          default:
            FAIL_EXIT1 ("invalid address family %d", addr->sa_family);
          }

        char host[NI_MAXHOST];
        char service[NI_MAXSERV];
        int ret = getnameinfo (addr, addrlen,
                               host, sizeof (host), service, sizeof (service),
                               NI_NUMERICHOST | NI_NUMERICSERV);

        if (ret != 0)
          {
            if (ret == EAI_SYSTEM)
              fprintf (fp, "; error: getnameinfo: %m\n");
            else
              fprintf (fp, "; error: getnameinfo: %s\n", gai_strerror (ret));
          }
        else
          fprintf (fp, "; nameserver[%zu]: [%s]:%s\n", i, host, service);
        ++i;
      }
  }

  TEST_VERIFY (!ferror (fp));

  __resolv_context_put (ctx);
}

/* Parameters of one test case.  */
struct test_case
{
  /* A short, descriptive name of the test.  */
  const char *name;

  /* The contents of the /etc/resolv.conf file.  */
  const char *conf;

  /* The expected output from print_resp.  */
  const char *expected;

  /* Setting for the LOCALDOMAIN environment variable.  NULL if the
     variable is not to be set.  */
  const char *localdomain;

  /* Setting for the RES_OPTIONS environment variable.  NULL if the
     variable is not to be set.  */
  const char *res_options;

  /* Override the system host name.  NULL means that no change is made
     and the default is used (test_hostname).  */
  const char *hostname;
};

enum test_init
{
  test_init,
  test_ninit,
  test_mkquery,
  test_gethostbyname,
  test_getaddrinfo,
  test_init_method_last = test_getaddrinfo
};

static const char *const test_init_names[] =
  {
    [test_init] = "res_init",
    [test_ninit] = "res_ninit",
    [test_mkquery] = "res_mkquery",
    [test_gethostbyname] = "gethostbyname",
    [test_getaddrinfo] = "getaddrinfo",
  };

/* Closure argument for run_res_init.  */
struct test_context
{
  enum test_init init;
  const struct test_case *t;
};

static void
setup_nss_dns_and_chroot (void)
{
  /* Load nss_dns outside of the chroot.  */
  if (dlopen (LIBNSS_DNS_SO, RTLD_LAZY) == NULL)
    FAIL_EXIT1 ("could not load " LIBNSS_DNS_SO ": %s", dlerror ());
  xchroot (chroot_env->path_chroot);
  /* Force the use of nss_dns.  */
  __nss_configure_lookup ("hosts", "dns");
}

/* Run res_ninit or res_init in a subprocess and dump the parsed
   resolver state to standard output.  */
static void
run_res_init (void *closure)
{
  struct test_context *ctx = closure;
  TEST_VERIFY (getenv ("LOCALDOMAIN") == NULL);
  TEST_VERIFY (getenv ("RES_OPTIONS") == NULL);
  if (ctx->t->localdomain != NULL)
    setenv ("LOCALDOMAIN", ctx->t->localdomain, 1);
  if (ctx->t->res_options != NULL)
    setenv ("RES_OPTIONS", ctx->t->res_options, 1);
  if (ctx->t->hostname != NULL)
    {
#ifdef CLONE_NEWUTS
      /* This test needs its own namespace, to avoid changing the host
         name for the parent, too.  */
      TEST_VERIFY_EXIT (unshare (CLONE_NEWUTS) == 0);
      if (sethostname (ctx->t->hostname, strlen (ctx->t->hostname)) != 0)
        FAIL_EXIT1 ("sethostname (\"%s\"): %m", ctx->t->hostname);
#else
      FAIL_UNSUPPORTED ("clone (CLONE_NEWUTS) not supported");
#endif
    }

  switch (ctx->init)
    {
    case test_init:
      xchroot (chroot_env->path_chroot);
      TEST_VERIFY (res_init () == 0);
      print_resp (stdout, &_res);
      return;

    case test_ninit:
      xchroot (chroot_env->path_chroot);
      res_state resp = xmalloc (sizeof (*resp));
      memset (resp, 0, sizeof (*resp));
      TEST_VERIFY (res_ninit (resp) == 0);
      print_resp (stdout, resp);
      res_nclose (resp);
      free (resp);
      return;

    case test_mkquery:
      xchroot (chroot_env->path_chroot);
      unsigned char buf[512];
      TEST_VERIFY (res_mkquery (QUERY, "www.example",
                                C_IN, ns_t_a, NULL, 0,
                                NULL, buf, sizeof (buf)) > 0);
      print_resp (stdout, &_res);
      return;

    case test_gethostbyname:
      setup_nss_dns_and_chroot ();
      /* Trigger implicit initialization of the _res structure.  The
         actual lookup result is immaterial.  */
      (void )gethostbyname ("www.example");
      print_resp (stdout, &_res);
      return;

    case test_getaddrinfo:
      setup_nss_dns_and_chroot ();
      /* Trigger implicit initialization of the _res structure.  The
         actual lookup result is immaterial.  */
      struct addrinfo *ai;
      (void) getaddrinfo ("www.example", NULL, NULL, &ai);
      print_resp (stdout, &_res);
      return;
    }

  FAIL_EXIT1 ("invalid init method %d", ctx->init);
}

#if TEST_THREAD
/* Helper function which calls run_res_init from a thread.  */
static void *
run_res_init_thread_func (void *closure)
{
  run_res_init (closure);
  return NULL;
}

/* Variant of res_run_init which runs the function on a non-main
   thread.  */
static void
run_res_init_on_thread (void *closure)
{
  xpthread_join (xpthread_create (NULL, run_res_init_thread_func, closure));
}
#endif /* TEST_THREAD */

struct test_case test_cases[] =
  {
    {.name = "empty file",
     .conf = "",
     .expected = "search example.com\n"
     "; search[0]: example.com\n"
     "nameserver 127.0.0.1\n"
     "; nameserver[0]: [127.0.0.1]:53\n"
    },
    {.name = "empty file, no-dot hostname",
     .conf = "",
     .expected = "nameserver 127.0.0.1\n"
     "; nameserver[0]: [127.0.0.1]:53\n",
     .hostname = "example",
    },
    {.name = "empty file with LOCALDOMAIN",
     .conf = "",
     .expected = "search example.net\n"
     "; search[0]: example.net\n"
     "nameserver 127.0.0.1\n"
     "; nameserver[0]: [127.0.0.1]:53\n",
     .localdomain = "example.net",
    },
    {.name = "empty file with RES_OPTIONS",
     .conf = "",
     .expected = "options attempts:5 edns0\n"
     "search example.com\n"
     "; search[0]: example.com\n"
     "nameserver 127.0.0.1\n"
     "; nameserver[0]: [127.0.0.1]:53\n",
     .res_options = "edns0 attempts:5",
    },
    {.name = "empty file with RES_OPTIONS and LOCALDOMAIN",
     .conf = "",
     .expected = "options attempts:5 edns0\n"
     "search example.org\n"
     "; search[0]: example.org\n"
     "nameserver 127.0.0.1\n"
     "; nameserver[0]: [127.0.0.1]:53\n",
     .localdomain = "example.org",
     .res_options = "edns0 attempts:5",
    },
    {.name = "basic",
     .conf =  "search corp.example.com example.com\n"
     "nameserver 192.0.2.1\n",
     .expected = "search corp.example.com example.com\n"
     "; search[0]: corp.example.com\n"
     "; search[1]: example.com\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
    },
    {.name = "basic with no-dot hostname",
     .conf = "search corp.example.com example.com\n"
     "nameserver 192.0.2.1\n",
     .expected = "search corp.example.com example.com\n"
     "; search[0]: corp.example.com\n"
     "; search[1]: example.com\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n",
     .hostname = "example",
    },
    {.name = "basic no-reload",
     .conf = "options no-reload\n"
     "search corp.example.com example.com\n"
     "nameserver 192.0.2.1\n",
     .expected = "options no-reload\n"
     "search corp.example.com example.com\n"
     "; search[0]: corp.example.com\n"
     "; search[1]: example.com\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
    },
    {.name = "basic no-reload via RES_OPTIONS",
     .conf = "search corp.example.com example.com\n"
     "nameserver 192.0.2.1\n",
     .expected = "options no-reload\n"
     "search corp.example.com example.com\n"
     "; search[0]: corp.example.com\n"
     "; search[1]: example.com\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n",
     .res_options = "no-reload"
    },
    {.name = "whitespace",
     .conf = "# This test covers comment and whitespace processing "
     " (trailing whitespace,\n"
     "# missing newline at end of file).\n"
     "\n"
     ";search commented out\n"
     "search corp.example.com\texample.com \n"
     "#nameserver 192.0.2.3\n"
     "nameserver 192.0.2.1 \n"
     "nameserver 192.0.2.2",    /* No \n at end of file.  */
     .expected = "search corp.example.com example.com\n"
     "; search[0]: corp.example.com\n"
     "; search[1]: example.com\n"
     "nameserver 192.0.2.1\n"
     "nameserver 192.0.2.2\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
     "; nameserver[1]: [192.0.2.2]:53\n"
    },
    {.name = "domain",
     .conf = "domain example.net\n"
     "nameserver 192.0.2.1\n",
     .expected = "search example.net\n"
     "; search[0]: example.net\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
    },
    {.name = "domain space",
     .conf = "domain example.net \n"
     "nameserver 192.0.2.1\n",
     .expected = "search example.net\n"
     "; search[0]: example.net\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
    },
    {.name = "domain tab",
     .conf = "domain example.net\t\n"
     "nameserver 192.0.2.1\n",
     .expected = "search example.net\n"
     "; search[0]: example.net\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
    },
    {.name = "domain override",
     .conf = "search example.com example.org\n"
     "nameserver 192.0.2.1\n"
     "domain example.net",      /* No \n at end of file.  */
     .expected = "search example.net\n"
     "; search[0]: example.net\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
    },
    {.name = "option values, multiple servers",
     .conf = "options\tinet6\tndots:3 edns0\tattempts:5\ttimeout:19\n"
     "domain  example.net\n"
     ";domain comment\n"
     "search corp.example.com\texample.com\n"
     "nameserver 192.0.2.1\n"
     "nameserver ::1\n"
     "nameserver 192.0.2.2\n",
     .expected = "options ndots:3 timeout:19 attempts:5 edns0\n"
     "search corp.example.com example.com\n"
     "; search[0]: corp.example.com\n"
     "; search[1]: example.com\n"
     "nameserver 192.0.2.1\n"
     "nameserver ::1\n"
     "nameserver 192.0.2.2\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
     "; nameserver[1]: [::1]:53\n"
     "; nameserver[2]: [192.0.2.2]:53\n"
    },
    {.name = "out-of-range option vales",
     .conf = "options use-vc timeout:999 attempts:999 ndots:99\n"
     "search example.com\n",
     .expected = "options ndots:15 timeout:30 attempts:5 use-vc\n"
     "search example.com\n"
     "; search[0]: example.com\n"
     "nameserver 127.0.0.1\n"
     "; nameserver[0]: [127.0.0.1]:53\n"
    },
    {.name = "repeated directives",
     .conf = "options ndots:3 use-vc\n"
     "options edns0 ndots:2\n"
     "domain corp.example\n"
     "search example.net corp.example.com example.com\n"
     "search example.org\n"
     "search\n",
     .expected = "options ndots:2 use-vc edns0\n"
     "search example.org\n"
     "; search[0]: example.org\n"
     "nameserver 127.0.0.1\n"
     "; nameserver[0]: [127.0.0.1]:53\n"
    },
    {.name = "many name servers, sortlist",
     .conf = "options single-request\n"
     "search example.org example.com example.net corp.example.com\n"
     "sortlist 192.0.2.0/255.255.255.0\n"
     "nameserver 192.0.2.1\n"
     "nameserver 192.0.2.2\n"
     "nameserver 192.0.2.3\n"
     "nameserver 192.0.2.4\n"
     "nameserver 192.0.2.5\n"
     "nameserver 192.0.2.6\n"
     "nameserver 192.0.2.7\n"
     "nameserver 192.0.2.8\n",
     .expected = "options single-request\n"
     "search example.org example.com example.net corp.example.com\n"
     "; search[0]: example.org\n"
     "; search[1]: example.com\n"
     "; search[2]: example.net\n"
     "; search[3]: corp.example.com\n"
     "sortlist 192.0.2.0/255.255.255.0\n"
     "nameserver 192.0.2.1\n"
     "nameserver 192.0.2.2\n"
     "nameserver 192.0.2.3\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
     "; nameserver[1]: [192.0.2.2]:53\n"
     "; nameserver[2]: [192.0.2.3]:53\n"
     "; nameserver[3]: [192.0.2.4]:53\n"
     "; nameserver[4]: [192.0.2.5]:53\n"
     "; nameserver[5]: [192.0.2.6]:53\n"
     "; nameserver[6]: [192.0.2.7]:53\n"
     "; nameserver[7]: [192.0.2.8]:53\n"
    },
    {.name = "IPv4 and IPv6 nameservers",
     .conf = "options single-request\n"
     "search example.org example.com example.net corp.example.com"
     " legacy.example.com\n"
     "sortlist 192.0.2.0\n"
     "nameserver 192.0.2.1\n"
     "nameserver 2001:db8::2\n"
     "nameserver 192.0.2.3\n"
     "nameserver 2001:db8::4\n"
     "nameserver 192.0.2.5\n"
     "nameserver 2001:db8::6\n"
     "nameserver 192.0.2.7\n"
     "nameserver 2001:db8::8\n",
     .expected = "options single-request\n"
     "search example.org example.com example.net corp.example.com"
     " legacy.example.com\n"
     "; search[0]: example.org\n"
     "; search[1]: example.com\n"
     "; search[2]: example.net\n"
     "; search[3]: corp.example.com\n"
     "; search[4]: legacy.example.com\n"
     "sortlist 192.0.2.0/255.255.255.0\n"
     "nameserver 192.0.2.1\n"
     "nameserver 2001:db8::2\n"
     "nameserver 192.0.2.3\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
     "; nameserver[1]: [2001:db8::2]:53\n"
     "; nameserver[2]: [192.0.2.3]:53\n"
     "; nameserver[3]: [2001:db8::4]:53\n"
     "; nameserver[4]: [192.0.2.5]:53\n"
     "; nameserver[5]: [2001:db8::6]:53\n"
     "; nameserver[6]: [192.0.2.7]:53\n"
     "; nameserver[7]: [2001:db8::8]:53\n",
    },
    {.name = "garbage after nameserver",
     .conf = "nameserver 192.0.2.1 garbage\n"
     "nameserver 192.0.2.2:5353\n"
     "nameserver 192.0.2.3 5353\n",
     .expected = "search example.com\n"
     "; search[0]: example.com\n"
     "nameserver 192.0.2.1\n"
     "nameserver 192.0.2.3\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
     "; nameserver[1]: [192.0.2.3]:53\n"
    },
    {.name = "RES_OPTIONS is cummulative",
     .conf = "options timeout:7 ndots:2 use-vc\n"
     "nameserver 192.0.2.1\n",
     .expected = "options ndots:3 timeout:7 attempts:5 use-vc edns0\n"
     "search example.com\n"
     "; search[0]: example.com\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n",
     .res_options = "attempts:5 ndots:3 edns0 ",
    },
    {.name = "many search list entries (bug 19569)",
     .conf = "nameserver 192.0.2.1\n"
     "search corp.example.com support.example.com"
     " community.example.org wan.example.net vpn.example.net"
     " example.com example.org example.net\n",
     .expected = "search corp.example.com support.example.com"
     " community.example.org wan.example.net vpn.example.net example.com\n"
     "; search[0]: corp.example.com\n"
     "; search[1]: support.example.com\n"
     "; search[2]: community.example.org\n"
     "; search[3]: wan.example.net\n"
     "; search[4]: vpn.example.net\n"
     "; search[5]: example.com\n"
     "; search[6]: example.org\n"
     "; search[7]: example.net\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
    },
    {.name = "very long search list entries (bug 21475)",
     .conf = "nameserver 192.0.2.1\n"
     "search example.com "
#define H63 "this-host-name-is-longer-than-yours-yes-I-really-really-mean-it"
#define D63 "this-domain-name-is-as-long-as-the-previous-name--63-characters"
     " " H63 "." D63 ".example.org"
     " " H63 "." D63 ".example.net\n",
     .expected = "search example.com " H63 "." D63 ".example.org\n"
     "; search[0]: example.com\n"
     "; search[1]: " H63 "." D63 ".example.org\n"
     "; search[2]: " H63 "." D63 ".example.net\n"
#undef H63
#undef D63
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
    },
    {.name = "trust-ad flag",
     .conf = "options trust-ad\n"
     "nameserver 192.0.2.1\n",
     .expected = "options trust-ad\n"
     "search example.com\n"
     "; search[0]: example.com\n"
     "nameserver 192.0.2.1\n"
     "; nameserver[0]: [192.0.2.1]:53\n"
    },
    { NULL }
  };

/* Run the indicated test case.  This function assumes that the chroot
   contents has already been set up.  */
static void
test_file_contents (const struct test_case *t)
{
#if TEST_THREAD
  for (int do_thread = 0; do_thread < 2; ++do_thread)
#endif
    for (int init_method = 0; init_method <= test_init_method_last;
         ++init_method)
      {
        if (test_verbose > 0)
          printf ("info:  testing init method %s\n",
                  test_init_names[init_method]);
        struct test_context ctx = { .init = init_method, .t = t };
        void (*func) (void *) = run_res_init;
#if TEST_THREAD
        if (do_thread)
          func = run_res_init_on_thread;
#endif
        struct support_capture_subprocess proc
          = support_capture_subprocess (func, &ctx);
        if (strcmp (proc.out.buffer, t->expected) != 0)
          {
            support_record_failure ();
            printf ("error: output mismatch for %s (init method %s)\n",
                    t->name, test_init_names[init_method]);
            support_run_diff ("expected", t->expected,
                              "actual", proc.out.buffer);
          }
        support_capture_subprocess_check (&proc, t->name, 0,
                                          sc_allow_stdout);
        support_capture_subprocess_free (&proc);
      }
}

/* Special tests which do not follow the general pattern.  */
enum { special_tests_count = 11 };

/* Implementation of special tests.  */
static void
special_test_callback (void *closure)
{
  unsigned int *test_indexp = closure;
  unsigned test_index = *test_indexp;
  TEST_VERIFY (test_index < special_tests_count);
  if (test_verbose > 0)
    printf ("info: special test %u\n", test_index);
  xchroot (chroot_env->path_chroot);

  switch (test_index)
    {
    case 0:
    case 1:
      /* Second res_init with missing or empty file preserves
         flags.  */
      if (test_index == 1)
        TEST_VERIFY (unlink (_PATH_RESCONF) == 0);
      _res.options = RES_USE_EDNS0;
      TEST_VERIFY (res_init () == 0);
      /* First res_init clears flag.  */
      TEST_VERIFY (!(_res.options & RES_USE_EDNS0));
      _res.options |= RES_USE_EDNS0;
      TEST_VERIFY (res_init () == 0);
      /* Second res_init preserves flag.  */
      TEST_VERIFY (_res.options & RES_USE_EDNS0);
      if (test_index == 1)
        /* Restore empty file.  */
        support_write_file_string (_PATH_RESCONF, "");
      break;

    case 2:
      /* Second res_init is cumulative.  */
      support_write_file_string (_PATH_RESCONF,
                                 "options rotate\n"
                                 "nameserver 192.0.2.1\n");
      _res.options = RES_USE_EDNS0;
      TEST_VERIFY (res_init () == 0);
      /* First res_init clears flag.  */
      TEST_VERIFY (!(_res.options & RES_USE_EDNS0));
      /* And sets RES_ROTATE.  */
      TEST_VERIFY (_res.options & RES_ROTATE);
      _res.options |= RES_USE_EDNS0;
      TEST_VERIFY (res_init () == 0);
      /* Second res_init preserves flag.  */
      TEST_VERIFY (_res.options & RES_USE_EDNS0);
      TEST_VERIFY (_res.options & RES_ROTATE);
      /* Reloading the configuration does not clear the explicitly set
         flag.  */
      support_write_file_string (_PATH_RESCONF,
                                 "nameserver 192.0.2.1\n"
                                 "nameserver 192.0.2.2\n");
      TEST_VERIFY (res_init () == 0);
      TEST_VERIFY (_res.nscount == 2);
      TEST_VERIFY (_res.options & RES_USE_EDNS0);
      /* Whether RES_ROTATE (originally in resolv.conf, now removed)
         should be preserved is subject to debate.  See bug 21701.  */
      /* TEST_VERIFY (!(_res.options & RES_ROTATE)); */
      break;

    case 3:
    case 4:
    case 5:
    case 6:
      support_write_file_string (_PATH_RESCONF,
                                 "options edns0\n"
                                 "nameserver 192.0.2.1\n");
      goto reload_tests;
    case 7: /* 7 and the following tests are with no-reload.  */
    case 8:
    case 9:
    case 10:
        support_write_file_string (_PATH_RESCONF,
                                   "options edns0 no-reload\n"
                                   "nameserver 192.0.2.1\n");
        /* Fall through.  */
    reload_tests:
      for (int iteration = 0; iteration < 2; ++iteration)
        {
          switch (test_index)
            {
            case 3:
            case 7:
              TEST_VERIFY (res_init () == 0);
              break;
            case 4:
            case 8:
              {
                unsigned char buf[512];
                TEST_VERIFY
                  (res_mkquery (QUERY, test_hostname, C_IN, T_A,
                                NULL, 0, NULL, buf, sizeof (buf)) > 0);
              }
              break;
            case 5:
            case 9:
              gethostbyname (test_hostname);
              break;
            case 6:
            case 10:
              {
                struct addrinfo *ai;
                (void) getaddrinfo (test_hostname, NULL, NULL, &ai);
              }
              break;
            }
          /* test_index == 7 is res_init and performs a reload even
             with no-reload.  */
          if (iteration == 0 || test_index > 7)
            {
              TEST_VERIFY (_res.options & RES_USE_EDNS0);
              TEST_VERIFY (!(_res.options & RES_ROTATE));
              if (test_index < 7)
                TEST_VERIFY (!(_res.options & RES_NORELOAD));
              else
                TEST_VERIFY (_res.options & RES_NORELOAD);
              TEST_VERIFY (_res.nscount == 1);
              /* File change triggers automatic reloading.  */
              support_write_file_string (_PATH_RESCONF,
                                         "options rotate\n"
                                         "nameserver 192.0.2.1\n"
                                         "nameserver 192.0.2.2\n");
            }
          else
            {
              if (test_index != 3 && test_index != 7)
                /* test_index 3, 7 are res_init; this function does
                   not reset flags.  See bug 21701.  */
                TEST_VERIFY (!(_res.options & RES_USE_EDNS0));
              TEST_VERIFY (_res.options & RES_ROTATE);
              TEST_VERIFY (_res.nscount == 2);
            }
        }
      break;
    }
}

#if TEST_THREAD
/* Helper function which calls special_test_callback from a
   thread.  */
static void *
special_test_thread_func (void *closure)
{
  special_test_callback (closure);
  return NULL;
}

/* Variant of special_test_callback which runs the function on a
   non-main thread.  */
static void
run_special_test_on_thread (void *closure)
{
  xpthread_join (xpthread_create (NULL, special_test_thread_func, closure));
}
#endif /* TEST_THREAD */

/* Perform the requested special test in a subprocess using
   special_test_callback.  */
static void
special_test (unsigned int test_index)
{
#if TEST_THREAD
  for (int do_thread = 0; do_thread < 2; ++do_thread)
#endif
    {
      void (*func) (void *) = special_test_callback;
#if TEST_THREAD
      if (do_thread)
        func = run_special_test_on_thread;
#endif
      struct support_capture_subprocess proc
        = support_capture_subprocess (func, &test_index);
      char *test_name = xasprintf ("special test %u", test_index);
      if (strcmp (proc.out.buffer, "") != 0)
        {
          support_record_failure ();
          printf ("error: output mismatch for %s\n", test_name);
          support_run_diff ("expected", "",
                            "actual", proc.out.buffer);
        }
      support_capture_subprocess_check (&proc, test_name, 0, sc_allow_stdout);
      free (test_name);
      support_capture_subprocess_free (&proc);
    }
}


/* Dummy DNS server.  It ensures that the probe queries sent by
   gethostbyname and getaddrinfo receive a reply even if the system
   applies a very strict rate limit to localhost.  */
static pid_t
start_dummy_server (void)
{
  int server_socket = xsocket (AF_INET, SOCK_DGRAM, 0);
  {
    struct sockaddr_in sin =
      {
        .sin_family = AF_INET,
        .sin_addr = { .s_addr = htonl (INADDR_LOOPBACK) },
        .sin_port = htons (53),
      };
    int ret = bind (server_socket, (struct sockaddr *) &sin, sizeof (sin));
    if (ret < 0)
      {
        if (errno == EACCES)
          /* The port is reserved, which means we cannot start the
             server.  */
          return -1;
        FAIL_EXIT1 ("cannot bind socket to port 53: %m");
      }
  }

  pid_t pid = xfork ();
  if (pid == 0)
    {
      /* Child process.  Echo back queries as SERVFAIL responses.  */
      while (true)
        {
          union
          {
            HEADER header;
            unsigned char bytes[512];
          } packet;
          struct sockaddr_in sin;
          socklen_t sinlen = sizeof (sin);

          ssize_t ret = recvfrom
            (server_socket, &packet, sizeof (packet),
             MSG_NOSIGNAL, (struct sockaddr *) &sin, &sinlen);
          if (ret < 0)
            FAIL_EXIT1 ("recvfrom on fake server socket: %m");
          if (ret > sizeof (HEADER))
            {
              /* Turn the query into a SERVFAIL response.  */
              packet.header.qr = 1;
              packet.header.rcode = ns_r_servfail;

              /* Send the response.  */
              ret = sendto (server_socket, &packet, ret,
                            MSG_NOSIGNAL, (struct sockaddr *) &sin, sinlen);
              if (ret < 0)
                /* The peer may have closed socket prematurely, so
                   this is not an error.  */
                printf ("warning: sending DNS server reply: %m\n");
            }
        }
    }

  /* In the parent, close the socket.  */
  xclose (server_socket);

  return pid;
}

static int
do_test (void)
{
  support_become_root ();
  support_enter_network_namespace ();
  if (!support_in_uts_namespace () || !support_can_chroot ())
    return EXIT_UNSUPPORTED;

  /* We are in an UTS namespace, so we can set the host name without
     altering the state of the entire system.  */
  if (sethostname (test_hostname, strlen (test_hostname)) != 0)
    FAIL_EXIT1 ("sethostname: %m");

  /* These environment variables affect resolv.conf parsing.  */
  unsetenv ("LOCALDOMAIN");
  unsetenv ("RES_OPTIONS");

  /* Ensure that the chroot setup worked.  */
  {
    struct support_capture_subprocess proc
      = support_capture_subprocess (check_chroot_working, NULL);
    support_capture_subprocess_check (&proc, "chroot", 0, sc_allow_none);
    support_capture_subprocess_free (&proc);
  }

  pid_t server = start_dummy_server ();

  for (size_t i = 0; test_cases[i].name != NULL; ++i)
    {
      if (test_verbose > 0)
        printf ("info: running test: %s\n", test_cases[i].name);
      TEST_VERIFY (test_cases[i].conf != NULL);
      TEST_VERIFY (test_cases[i].expected != NULL);

      support_write_file_string (chroot_env->path_resolv_conf,
                                 test_cases[i].conf);

      test_file_contents (&test_cases[i]);

      /* The expected output from the empty file test is used for
         further tests.  */
      if (test_cases[i].conf[0] == '\0')
        {
          if (test_verbose > 0)
            printf ("info:  special test: missing file\n");
          TEST_VERIFY (unlink (chroot_env->path_resolv_conf) == 0);
          test_file_contents (&test_cases[i]);

          if (test_verbose > 0)
            printf ("info:  special test: dangling symbolic link\n");
          TEST_VERIFY (symlink ("does-not-exist", chroot_env->path_resolv_conf) == 0);
          test_file_contents (&test_cases[i]);
          TEST_VERIFY (unlink (chroot_env->path_resolv_conf) == 0);

          if (test_verbose > 0)
            printf ("info:  special test: unreadable file\n");
          support_write_file_string (chroot_env->path_resolv_conf, "");
          TEST_VERIFY (chmod (chroot_env->path_resolv_conf, 0) == 0);
          test_file_contents (&test_cases[i]);

          /* Restore the empty file.  */
          TEST_VERIFY (unlink (chroot_env->path_resolv_conf) == 0);
          support_write_file_string (chroot_env->path_resolv_conf, "");
        }
    }

  /* The tests which do not follow a regular pattern.  */
  for (unsigned int test_index = 0;
       test_index < special_tests_count; ++test_index)
    special_test (test_index);

  if (server > 0)
    {
      if (kill (server, SIGTERM) < 0)
        FAIL_EXIT1 ("could not terminate server process: %m");
      xwaitpid (server, NULL, 0);
    }

  support_chroot_free (chroot_env);
  return 0;
}

#define PREPARE prepare
#include <support/test-driver.c>
