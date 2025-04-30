/*
 * clnt_perror.c
 *
 * Copyright (c) 2010, 2011, Oracle America, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the "Oracle America, Inc." nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 *   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <string.h>
#include <libintl.h>
#include <rpc/rpc.h>
#include <wchar.h>
#include <libio/iolibio.h>
#include <shlib-compat.h>

static char *auth_errmsg (enum auth_stat stat);

/*
 * Making buf a preprocessor macro requires renaming the local
 * buf variable in a few functions.  Overriding a global variable
 * with a local variable of the same name is a bad idea, anyway.
 */
#define buf RPC_THREAD_VARIABLE(clnt_perr_buf_s)

/*
 * Print reply error info
 */
char *
clnt_sperror (CLIENT * rpch, const char *msg)
{
  struct rpc_err e;
  CLNT_GETERR (rpch, &e);

  const char *errstr = clnt_sperrno (e.re_status);

  char chrbuf[1024];
  char *str;
  char *tmpstr;
  int res;
  switch (e.re_status)
    {
    case RPC_SUCCESS:
    case RPC_CANTENCODEARGS:
    case RPC_CANTDECODERES:
    case RPC_TIMEDOUT:
    case RPC_PROGUNAVAIL:
    case RPC_PROCUNAVAIL:
    case RPC_CANTDECODEARGS:
    case RPC_SYSTEMERROR:
    case RPC_UNKNOWNHOST:
    case RPC_UNKNOWNPROTO:
    case RPC_PMAPFAILURE:
    case RPC_PROGNOTREGISTERED:
    case RPC_FAILED:
      res = __asprintf (&str, "%s: %s\n", msg, errstr);
      break;

    case RPC_CANTSEND:
    case RPC_CANTRECV:
      res = __asprintf (&str, "%s: %s; errno = %s\n",
			msg, errstr, __strerror_r (e.re_errno,
						   chrbuf, sizeof chrbuf));
      break;

    case RPC_VERSMISMATCH:
      res = __asprintf (&str,
			_("%s: %s; low version = %lu, high version = %lu"),
			msg, errstr, e.re_vers.low, e.re_vers.high);
      break;

    case RPC_AUTHERROR:
      tmpstr = auth_errmsg (e.re_why);
      if (tmpstr != NULL)
	res = __asprintf (&str, _("%s: %s; why = %s\n"), msg, errstr, tmpstr);
      else
	res = __asprintf (&str, _("\
%s: %s; why = (unknown authentication error - %d)\n"),
			  msg, errstr, (int) e.re_why);
      break;

    case RPC_PROGVERSMISMATCH:
      res = __asprintf (&str,
			_("%s: %s; low version = %lu, high version = %lu"),
			msg, errstr, e.re_vers.low, e.re_vers.high);
      break;

    default:			/* unknown */
      res = __asprintf (&str, "%s: %s; s1 = %lu, s2 = %lu",
			msg, errstr, e.re_lb.s1, e.re_lb.s2);
      break;
    }

  if (res < 0)
    return NULL;

  char *oldbuf = buf;
  buf = str;
  free (oldbuf);

  return str;
}
libc_hidden_nolink_sunrpc (clnt_sperror, GLIBC_2_0)

void
clnt_perror (CLIENT * rpch, const char *msg)
{
  (void) __fxprintf (NULL, "%s", clnt_sperror (rpch, msg));
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (clnt_perror)
#else
libc_hidden_nolink_sunrpc (clnt_perror, GLIBC_2_0)
#endif


struct rpc_errtab
{
  enum clnt_stat status;
  unsigned int message_off;
};

static const char rpc_errstr[] =
{
#define RPC_SUCCESS_IDX		0
  N_("RPC: Success")
  "\0"
#define RPC_CANTENCODEARGS_IDX	(RPC_SUCCESS_IDX + sizeof "RPC: Success")
  N_("RPC: Can't encode arguments")
  "\0"
#define RPC_CANTDECODERES_IDX	(RPC_CANTENCODEARGS_IDX \
				 + sizeof "RPC: Can't encode arguments")
  N_("RPC: Can't decode result")
  "\0"
#define RPC_CANTSEND_IDX	(RPC_CANTDECODERES_IDX \
				 + sizeof "RPC: Can't decode result")
  N_("RPC: Unable to send")
  "\0"
#define RPC_CANTRECV_IDX	(RPC_CANTSEND_IDX \
				 + sizeof "RPC: Unable to send")
  N_("RPC: Unable to receive")
  "\0"
#define RPC_TIMEDOUT_IDX	(RPC_CANTRECV_IDX \
				 + sizeof "RPC: Unable to receive")
  N_("RPC: Timed out")
  "\0"
#define RPC_VERSMISMATCH_IDX	(RPC_TIMEDOUT_IDX \
				 + sizeof "RPC: Timed out")
  N_("RPC: Incompatible versions of RPC")
  "\0"
#define RPC_AUTHERROR_IDX	(RPC_VERSMISMATCH_IDX \
				 + sizeof "RPC: Incompatible versions of RPC")
  N_("RPC: Authentication error")
  "\0"
#define RPC_PROGUNAVAIL_IDX		(RPC_AUTHERROR_IDX \
				 + sizeof "RPC: Authentication error")
  N_("RPC: Program unavailable")
  "\0"
#define RPC_PROGVERSMISMATCH_IDX (RPC_PROGUNAVAIL_IDX \
				  + sizeof "RPC: Program unavailable")
  N_("RPC: Program/version mismatch")
  "\0"
#define RPC_PROCUNAVAIL_IDX	(RPC_PROGVERSMISMATCH_IDX \
				 + sizeof "RPC: Program/version mismatch")
  N_("RPC: Procedure unavailable")
  "\0"
#define RPC_CANTDECODEARGS_IDX	(RPC_PROCUNAVAIL_IDX \
				 + sizeof "RPC: Procedure unavailable")
  N_("RPC: Server can't decode arguments")
  "\0"
#define RPC_SYSTEMERROR_IDX	(RPC_CANTDECODEARGS_IDX \
				 + sizeof "RPC: Server can't decode arguments")
  N_("RPC: Remote system error")
  "\0"
#define RPC_UNKNOWNHOST_IDX	(RPC_SYSTEMERROR_IDX \
				 + sizeof "RPC: Remote system error")
  N_("RPC: Unknown host")
  "\0"
#define RPC_UNKNOWNPROTO_IDX	(RPC_UNKNOWNHOST_IDX \
				 + sizeof "RPC: Unknown host")
  N_("RPC: Unknown protocol")
  "\0"
#define RPC_PMAPFAILURE_IDX	(RPC_UNKNOWNPROTO_IDX \
				 + sizeof "RPC: Unknown protocol")
  N_("RPC: Port mapper failure")
  "\0"
#define RPC_PROGNOTREGISTERED_IDX (RPC_PMAPFAILURE_IDX \
				   + sizeof "RPC: Port mapper failure")
  N_("RPC: Program not registered")
  "\0"
#define RPC_FAILED_IDX		(RPC_PROGNOTREGISTERED_IDX \
				 + sizeof "RPC: Program not registered")
  N_("RPC: Failed (unspecified error)")
};

static const struct rpc_errtab rpc_errlist[] =
{
  { RPC_SUCCESS, RPC_SUCCESS_IDX },
  { RPC_CANTENCODEARGS, RPC_CANTENCODEARGS_IDX },
  { RPC_CANTDECODERES, RPC_CANTDECODERES_IDX },
  { RPC_CANTSEND, RPC_CANTSEND_IDX },
  { RPC_CANTRECV, RPC_CANTRECV_IDX },
  { RPC_TIMEDOUT, RPC_TIMEDOUT_IDX },
  { RPC_VERSMISMATCH, RPC_VERSMISMATCH_IDX },
  { RPC_AUTHERROR, RPC_AUTHERROR_IDX },
  { RPC_PROGUNAVAIL, RPC_PROGUNAVAIL_IDX },
  { RPC_PROGVERSMISMATCH, RPC_PROGVERSMISMATCH_IDX },
  { RPC_PROCUNAVAIL, RPC_PROCUNAVAIL_IDX },
  { RPC_CANTDECODEARGS, RPC_CANTDECODEARGS_IDX },
  { RPC_SYSTEMERROR, RPC_SYSTEMERROR_IDX },
  { RPC_UNKNOWNHOST, RPC_UNKNOWNHOST_IDX },
  { RPC_UNKNOWNPROTO, RPC_UNKNOWNPROTO_IDX },
  { RPC_PMAPFAILURE, RPC_PMAPFAILURE_IDX },
  { RPC_PROGNOTREGISTERED, RPC_PROGNOTREGISTERED_IDX },
  { RPC_FAILED, RPC_FAILED_IDX }
};


/*
 * This interface for use by clntrpc
 */
char *
clnt_sperrno (enum clnt_stat stat)
{
  size_t i;

  for (i = 0; i < sizeof (rpc_errlist) / sizeof (struct rpc_errtab); i++)
    {
      if (rpc_errlist[i].status == stat)
	{
	  return _(rpc_errstr + rpc_errlist[i].message_off);
	}
    }
  return _("RPC: (unknown error code)");
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (clnt_sperrno)
#else
libc_hidden_nolink_sunrpc (clnt_sperrno, GLIBC_2_0)
#endif

void
clnt_perrno (enum clnt_stat num)
{
  (void) __fxprintf (NULL, "%s", clnt_sperrno (num));
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (clnt_perrno)
#else
libc_hidden_nolink_sunrpc (clnt_perrno, GLIBC_2_0)
#endif

char *
clnt_spcreateerror (const char *msg)
{
  struct rpc_createerr *ce = &get_rpc_createerr ();

  char chrbuf[1024];
  const char *connector = "";
  const char *errstr = "";
  switch (ce->cf_stat)
    {
    case RPC_PMAPFAILURE:
      connector = " - ";
      errstr = clnt_sperrno (ce->cf_error.re_status);
      break;

    case RPC_SYSTEMERROR:
      connector = " - ";
      errstr = __strerror_r (ce->cf_error.re_errno, chrbuf, sizeof chrbuf);
      break;

    default:
      break;
    }

  char *str;
  if (__asprintf (&str, "%s: %s%s%s\n",
		  msg, clnt_sperrno (ce->cf_stat), connector, errstr) < 0)
    return NULL;

  char *oldbuf = buf;
  buf = str;
  free (oldbuf);

  return str;
}
libc_hidden_nolink_sunrpc (clnt_spcreateerror, GLIBC_2_0)

void
clnt_pcreateerror (const char *msg)
{
  (void) __fxprintf (NULL, "%s", clnt_spcreateerror (msg));
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (clnt_pcreateerror)
#else
libc_hidden_nolink_sunrpc (clnt_pcreateerror, GLIBC_2_0)
#endif

struct auth_errtab
{
  enum auth_stat status;
  unsigned int message_off;
};

static const char auth_errstr[] =
{
#define AUTH_OK_IDX		0
   N_("Authentication OK")
   "\0"
#define AUTH_BADCRED_IDX	(AUTH_OK_IDX + sizeof "Authentication OK")
   N_("Invalid client credential")
   "\0"
#define AUTH_REJECTEDCRED_IDX	(AUTH_BADCRED_IDX \
				 + sizeof "Invalid client credential")
   N_("Server rejected credential")
   "\0"
#define AUTH_BADVERF_IDX	(AUTH_REJECTEDCRED_IDX \
				 + sizeof "Server rejected credential")
   N_("Invalid client verifier")
   "\0"
#define AUTH_REJECTEDVERF_IDX	(AUTH_BADVERF_IDX \
				 + sizeof "Invalid client verifier")
   N_("Server rejected verifier")
   "\0"
#define AUTH_TOOWEAK_IDX	(AUTH_REJECTEDVERF_IDX \
				 + sizeof "Server rejected verifier")
   N_("Client credential too weak")
   "\0"
#define AUTH_INVALIDRESP_IDX	(AUTH_TOOWEAK_IDX \
				 + sizeof "Client credential too weak")
   N_("Invalid server verifier")
   "\0"
#define AUTH_FAILED_IDX		(AUTH_INVALIDRESP_IDX \
				 + sizeof "Invalid server verifier")
   N_("Failed (unspecified error)")
};

static const struct auth_errtab auth_errlist[] =
{
  { AUTH_OK, AUTH_OK_IDX },
  { AUTH_BADCRED, AUTH_BADCRED_IDX },
  { AUTH_REJECTEDCRED, AUTH_REJECTEDCRED_IDX },
  { AUTH_BADVERF, AUTH_BADVERF_IDX },
  { AUTH_REJECTEDVERF, AUTH_REJECTEDVERF_IDX },
  { AUTH_TOOWEAK, AUTH_TOOWEAK_IDX },
  { AUTH_INVALIDRESP, AUTH_INVALIDRESP_IDX },
  { AUTH_FAILED, AUTH_FAILED_IDX }
};

static char *
auth_errmsg (enum auth_stat stat)
{
  size_t i;

  for (i = 0; i < sizeof (auth_errlist) / sizeof (struct auth_errtab); i++)
    {
      if (auth_errlist[i].status == stat)
	{
	  return _(auth_errstr + auth_errlist[i].message_off);
	}
    }
  return NULL;
}


libc_freeres_fn (free_mem)
{
  /* Not libc_freeres_ptr, since buf is a macro.  */
  free (buf);
}
