/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de>, 1997.

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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libintl.h>
#include <rpc/rpc.h>
#include <rpc/pmap_clnt.h>
#include <string.h>
#include <memory.h>
#include <syslog.h>
#include <sys/poll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <rpc/key_prot.h>
#include <rpcsvc/nis.h>
#include <rpcsvc/nis_callback.h>
#include <libc-lock.h>

#include "nis_xdr.h"
#include "nis_intern.h"

/* Sorry, we are not able to make this threadsafe. Stupid. But some
   functions doesn't send us a nis_result obj, so we don't have a
   cookie. Maybe we could use keys for threads ? Have to learn more
   about pthreads -- kukuk@vt.uni-paderborn.de */

static nis_cb *data;

__libc_lock_define_initialized (static, callback)


#if 0
static char *
__nis_getpkey(const char *sname)
{
  char buf[(strlen (sname) + 1) * 2 + 40];
  char pkey[HEXKEYBYTES + 1];
  char *cp, *domain;
  nis_result *res;
  unsigned int len = 0;

  domain = strchr (sname, '.');
  if (domain == NULL)
    return NULL;

  /* Remove prefixing dot */
  ++domain;

  cp = stpcpy (buf, "[cname=");
  cp = stpcpy (cp, sname);
  cp = stpcpy (cp, ",auth_type=DES],cred.org_dir.");
  cp = stpcpy (cp, domain);

  res = nis_list (buf, USE_DGRAM|NO_AUTHINFO|FOLLOW_LINKS|FOLLOW_PATH,
		  NULL, NULL);

  if (res == NULL)
    return NULL;

  if (NIS_RES_STATUS (res) != NIS_SUCCESS)
    {
      nis_freeresult (res);
      return NULL;
    }

  len = ENTRY_LEN(NIS_RES_OBJECT(res), 3);
  strncpy (pkey, ENTRY_VAL(NIS_RES_OBJECT(res), 3), len);
  pkey[len] = '\0';
  cp = strchr (pkey, ':');
  if (cp != NULL)
    *cp = '\0';

  nis_freeresult (res);

  return strdup (pkey);
}
#endif

static void
cb_prog_1 (struct svc_req *rqstp, SVCXPRT *transp)
{
  union
    {
      cback_data cbproc_receive_1_arg;
      nis_error cbproc_error_1_arg;
    }
  argument;
  char *result;
  xdrproc_t xdr_argument, xdr_result;
  bool_t bool_result;

  switch (rqstp->rq_proc)
    {
    case NULLPROC:
      svc_sendreply (transp, (xdrproc_t) xdr_void, (char *) NULL);
      return;

    case CBPROC_RECEIVE:
      {
	unsigned int i;

	xdr_argument = (xdrproc_t) xdr_cback_data;
	xdr_result = (xdrproc_t) xdr_bool;
	memset (&argument, 0, sizeof (argument));
	if (!svc_getargs (transp, xdr_argument, (caddr_t) & argument))
	  {
	    svcerr_decode (transp);
	    return;
	  }
	bool_result = FALSE;
	for (i = 0; i < argument.cbproc_receive_1_arg.entries.entries_len; ++i)
	  {
#define cbproc_entry(a) argument.cbproc_receive_1_arg.entries.entries_val[a]
	    char name[strlen (cbproc_entry(i)->zo_name)
		      + strlen (cbproc_entry(i)->zo_domain) + 3];
	    char *cp;

	    cp = stpcpy (name, cbproc_entry(i)->zo_name);
	    *cp++ = '.';
	    cp = stpcpy (cp, cbproc_entry(i)->zo_domain);

	    if ((data->callback) (name, cbproc_entry(i), data->userdata))
	      {
		bool_result = TRUE;
		data->nomore = 1;
		data->result = NIS_SUCCESS;
		break;
	      }
	  }
	result = (char *) &bool_result;
      }
      break;
    case CBPROC_FINISH:
      xdr_argument = (xdrproc_t) xdr_void;
      xdr_result = (xdrproc_t) xdr_void;
      memset (&argument, 0, sizeof (argument));
      if (!svc_getargs (transp, xdr_argument, (caddr_t) & argument))
	{
	  svcerr_decode (transp);
	  return;
	}
      data->nomore = 1;
      data->result = NIS_SUCCESS;
      bool_result = TRUE;	/* to make gcc happy, not necessary */
      result = (char *) &bool_result;
      break;
    case CBPROC_ERROR:
      xdr_argument = (xdrproc_t) _xdr_nis_error;
      xdr_result = (xdrproc_t) xdr_void;
      memset (&argument, 0, sizeof (argument));
      if (!svc_getargs (transp, xdr_argument, (caddr_t) & argument))
	{
	  svcerr_decode (transp);
	  return;
	}
      data->nomore = 1;
      data->result = argument.cbproc_error_1_arg;
      bool_result = TRUE;	/* to make gcc happy, not necessary */
      result = (char *) &bool_result;
      break;
    default:
      svcerr_noproc (transp);
      return;
    }
  if (result != NULL && !svc_sendreply (transp, xdr_result, result))
    svcerr_systemerr (transp);
  if (!svc_freeargs (transp, xdr_argument, (caddr_t) & argument))
    {
      fputs (_ ("unable to free arguments"), stderr);
      exit (1);
    }
  return;
}

static nis_error
internal_nis_do_callback (struct dir_binding *bptr, netobj *cookie,
			  struct nis_cb *cb)
{
  struct timeval TIMEOUT = {25, 0};
  bool_t cb_is_running;

  data = cb;

  for (;;)
    {
      struct pollfd my_pollfd[svc_max_pollfd];
      int i;

      if (svc_max_pollfd == 0 && svc_pollfd == NULL)
        return NIS_CBERROR;

      for (i = 0; i < svc_max_pollfd; ++i)
        {
          my_pollfd[i].fd = svc_pollfd[i].fd;
          my_pollfd[i].events = svc_pollfd[i].events;
          my_pollfd[i].revents = 0;
        }

      switch (i = TEMP_FAILURE_RETRY (__poll (my_pollfd, svc_max_pollfd,
					      25*1000)))
        {
	case -1:
	  return NIS_CBERROR;
	case 0:
	  /* See if callback 'thread' in the server is still alive. */
	  cb_is_running = FALSE;
	  if (clnt_call (bptr->clnt, NIS_CALLBACK, (xdrproc_t) xdr_netobj,
			 (caddr_t) cookie, (xdrproc_t) xdr_bool,
			 (caddr_t) &cb_is_running, TIMEOUT) != RPC_SUCCESS)
	    cb_is_running = FALSE;

	  if (cb_is_running == FALSE)
	    {
	      syslog (LOG_ERR, "NIS+: callback timed out");
	      return NIS_CBERROR;
	    }
	  break;
	default:
	  svc_getreq_poll (my_pollfd, i);
	  if (data->nomore)
	    return data->result;
	}
    }
}

nis_error
__nis_do_callback (struct dir_binding *bptr, netobj *cookie,
		   struct nis_cb *cb)
{
  nis_error result;

  __libc_lock_lock (callback);

  result = internal_nis_do_callback (bptr, cookie, cb);

  __libc_lock_unlock (callback);

  return result;
}

struct nis_cb *
__nis_create_callback (int (*callback) (const_nis_name, const nis_object *,
					const void *),
		       const void *userdata, unsigned int flags)
{
  struct nis_cb *cb;
  int sock = RPC_ANYSOCK;
  struct sockaddr_in sin;
  socklen_t len = sizeof (struct sockaddr_in);
  unsigned short port;
  int nomsg = 0;

  cb = (struct nis_cb *) calloc (1,
				 sizeof (struct nis_cb) + sizeof (nis_server));
  if (__glibc_unlikely (cb == NULL))
    goto failed;
  cb->serv = (nis_server *) (cb + 1);
  cb->serv->name = strdup (nis_local_principal ());
  if (__glibc_unlikely (cb->serv->name == NULL))
    goto failed;
  cb->serv->ep.ep_val = (endpoint *) calloc (2, sizeof (endpoint));
  if (__glibc_unlikely (cb->serv->ep.ep_val == NULL))
    goto failed;
  cb->serv->ep.ep_len = 1;
  cb->serv->ep.ep_val[0].family = strdup ("inet");
  if (__glibc_unlikely (cb->serv->ep.ep_val[0].family == NULL))
    goto failed;
  cb->callback = callback;
  cb->userdata = userdata;

  if ((flags & NO_AUTHINFO) || !key_secretkey_is_set ())
    {
      cb->serv->key_type = NIS_PK_NONE;
      cb->serv->pkey.n_bytes = NULL;
      cb->serv->pkey.n_len = 0;
    }
  else
    {
#if 0
      if ((cb->serv->pkey.n_bytes = __nis_getpkey (cb->serv->name)) == NULL)
	{
	  cb->serv->pkey.n_len = 0;
	  cb->serv->key_type = NIS_PK_NONE;
	}
      else
	{
	  cb->serv->key_type = NIS_PK_DH;
	  cb->serv->pkey.n_len = strlen(cb->serv->pkey.n_bytes);
	}
#else
      cb->serv->pkey.n_len =0;
      cb->serv->pkey.n_bytes = NULL;
      cb->serv->key_type = NIS_PK_NONE;
#endif
    }

  cb->serv->ep.ep_val[0].proto = strdup ((flags & USE_DGRAM) ? "udp" : "tcp");
  if (__glibc_unlikely (cb->serv->ep.ep_val[0].proto == NULL))
    goto failed;
  cb->xprt = ((flags & USE_DGRAM)
	      ? svcudp_bufcreate (sock, 100, 8192)
	      : svctcp_create (sock, 100, 8192));
  if (cb->xprt == NULL)
    {
      nomsg = 1;
      goto failed;
    }
  cb->sock = cb->xprt->xp_sock;
  if (!svc_register (cb->xprt, CB_PROG, CB_VERS, cb_prog_1, 0))
    {
      xprt_unregister (cb->xprt);
      svc_destroy (cb->xprt);
      xdr_free ((xdrproc_t) _xdr_nis_server, (char *) cb->serv);
      free (cb);
      syslog (LOG_ERR, "NIS+: failed to register callback dispatcher");
      return NULL;
    }

  if (getsockname (cb->sock, (struct sockaddr *) &sin, &len) == -1)
    {
      xprt_unregister (cb->xprt);
      svc_destroy (cb->xprt);
      xdr_free ((xdrproc_t) _xdr_nis_server, (char *) cb->serv);
      free (cb);
      syslog (LOG_ERR, "NIS+: failed to read local socket info");
      return NULL;
    }
  port = ntohs (sin.sin_port);
  get_myaddress (&sin);

  if (asprintf (&cb->serv->ep.ep_val[0].uaddr, "%s.%d.%d",
		inet_ntoa (sin.sin_addr), (port & 0xFF00) >> 8, port & 0x00FF)
      < 0)
    goto failed;

  return cb;

 failed:
  if (cb)
    {
      if (cb->xprt)
	svc_destroy (cb->xprt);
      xdr_free ((xdrproc_t) _xdr_nis_server, (char *) cb->serv);
      free (cb);
    }
  if (!nomsg)
    syslog (LOG_ERR, "NIS+: out of memory allocating callback");
  return NULL;
}

nis_error
__nis_destroy_callback (struct nis_cb *cb)
{
  xprt_unregister (cb->xprt);
  svc_destroy (cb->xprt);
  close (cb->sock);
  xdr_free ((xdrproc_t) _xdr_nis_server, (char *) cb->serv);
  free (cb);

  return NIS_SUCCESS;
}
