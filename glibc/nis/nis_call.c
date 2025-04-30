/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1997.

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
#include <fcntl.h>
#include <string.h>
#include <libintl.h>
#include <rpc/rpc.h>
#include <rpc/auth.h>
#include <rpcsvc/nis.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <libc-lock.h>

#include "nis_xdr.h"
#include "nis_intern.h"
#include <libnsl.h>
#include <shlib-compat.h>

static const struct timeval RPCTIMEOUT = {10, 0};
static const struct timeval UDPTIMEOUT = {5, 0};

extern u_short __pmap_getnisport (struct sockaddr_in *address, u_long program,
				  u_long version, u_int protocol);

unsigned long int
inetstr2int (const char *str)
{
  size_t j = 0;
  for (size_t i = 0; str[i] != '\0'; ++i)
    if (str[i] == '.' && __builtin_expect (++j == 4, 0))
      {
	char buffer[i + 1];
	buffer[i] = '\0';
	return inet_addr (memcpy (buffer, str, i));
      }

  return inet_addr (str);
}

void
__nisbind_destroy (dir_binding *bind)
{
  if (bind->clnt != NULL)
    {
      if (bind->use_auth)
	auth_destroy (bind->clnt->cl_auth);
      clnt_destroy (bind->clnt);
    }
}
libnsl_hidden_nolink_def (__nisbind_destroy, GLIBC_2_1)

nis_error
__nisbind_next (dir_binding *bind)
{
  if (bind->clnt != NULL)
    {
      if (bind->use_auth)
	auth_destroy (bind->clnt->cl_auth);
      clnt_destroy (bind->clnt);
      bind->clnt = NULL;
    }

  if (bind->trys >= bind->server_len)
    return NIS_FAIL;

  for (u_int j = bind->current_ep + 1;
       j < bind->server_val[bind->server_used].ep.ep_len; ++j)
    if (strcmp (bind->server_val[bind->server_used].ep.ep_val[j].family,
		"inet") == 0)
      if (bind->server_val[bind->server_used].ep.ep_val[j].proto[0] == '-')
	{
	  bind->current_ep = j;
	  return NIS_SUCCESS;
	}

  ++bind->trys;
  ++bind->server_used;
  if (bind->server_used >= bind->server_len)
    bind->server_used = 0;

  for (u_int j = 0; j < bind->server_val[bind->server_used].ep.ep_len; ++j)
    if (strcmp (bind->server_val[bind->server_used].ep.ep_val[j].family,
		"inet") == 0)
      if (bind->server_val[bind->server_used].ep.ep_val[j].proto[0] == '-')
	{
	  bind->current_ep = j;
	  return NIS_SUCCESS;
	}

  return NIS_FAIL;
}
libnsl_hidden_nolink_def (__nisbind_next, GLIBC_2_1)

static struct ckey_cache_entry
{
  struct in_addr inaddr;
  in_port_t port;
  unsigned int protocol;
  des_block ckey;
} *ckey_cache;
static size_t ckey_cache_size;
static size_t ckey_cache_allocated;
static pid_t ckey_cache_pid;
static uid_t ckey_cache_euid;
__libc_lock_define_initialized (static, ckey_cache_lock)

static bool_t
get_ckey (des_block *ckey, struct sockaddr_in *addr, unsigned int protocol)
{
  size_t i;
  pid_t pid = getpid ();
  uid_t euid = geteuid ();
  bool_t ret = FALSE;

  __libc_lock_lock (ckey_cache_lock);

  if (ckey_cache_pid != pid || ckey_cache_euid != euid)
    {
      ckey_cache_size = 0;
      ckey_cache_pid = pid;
      ckey_cache_euid = euid;
    }

  for (i = 0; i < ckey_cache_size; ++i)
    if (ckey_cache[i].port == addr->sin_port
	&& ckey_cache[i].protocol == protocol
	&& memcmp (&ckey_cache[i].inaddr, &addr->sin_addr,
		   sizeof (addr->sin_addr)) == 0)
      {
	*ckey = ckey_cache[i].ckey;
	ret = TRUE;
	break;
      }

  if (!ret && key_gendes (ckey) >= 0)
    {
      ret = TRUE;
      /* Don't grow the cache indefinitely.  */
      if (ckey_cache_size == 256)
	ckey_cache_size = 0;
      if (ckey_cache_size == ckey_cache_allocated)
	{
	  size_t size = ckey_cache_allocated ? ckey_cache_allocated * 2 : 16;
	  struct ckey_cache_entry *new_cache
	    = realloc (ckey_cache, size * sizeof (*ckey_cache));
	  if (new_cache != NULL)
	    {
	      ckey_cache = new_cache;
	      ckey_cache_allocated = size;
	    }
	}
      ckey_cache[ckey_cache_size].inaddr = addr->sin_addr;
      ckey_cache[ckey_cache_size].port = addr->sin_port;
      ckey_cache[ckey_cache_size].protocol = protocol;
      ckey_cache[ckey_cache_size++].ckey = *ckey;
    }

  __libc_lock_unlock (ckey_cache_lock);
  return ret;
}

nis_error
__nisbind_connect (dir_binding *dbp)
{
  nis_server *serv;
  u_short port;

  if (dbp == NULL)
    return NIS_FAIL;

  serv = &dbp->server_val[dbp->server_used];

  memset (&dbp->addr, '\0', sizeof (dbp->addr));
  dbp->addr.sin_family = AF_INET;

  dbp->addr.sin_addr.s_addr =
    inetstr2int (serv->ep.ep_val[dbp->current_ep].uaddr);

  if (dbp->addr.sin_addr.s_addr == INADDR_NONE)
    return NIS_FAIL;

  /* Check, if the host is online and rpc.nisd is running. Much faster
     then the clnt*_create functions: */
  port = __pmap_getnisport (&dbp->addr, NIS_PROG, NIS_VERSION,
			    dbp->use_udp ? IPPROTO_UDP : IPPROTO_TCP);
  if (port == 0)
    return NIS_RPCERROR;

  dbp->addr.sin_port = htons (port);
  dbp->socket = RPC_ANYSOCK;
  if (dbp->use_udp)
    dbp->clnt = clntudp_create (&dbp->addr, NIS_PROG, NIS_VERSION,
				 UDPTIMEOUT, &dbp->socket);
  else
    dbp->clnt = clnttcp_create (&dbp->addr, NIS_PROG, NIS_VERSION,
				 &dbp->socket, 0, 0);

  if (dbp->clnt == NULL)
    return NIS_RPCERROR;

  clnt_control (dbp->clnt, CLSET_TIMEOUT, (caddr_t) &RPCTIMEOUT);
  /* If the program exists, close the socket */
  if (fcntl (dbp->socket, F_SETFD, 1) == -1)
    perror ("fcntl: F_SETFD");

  if (dbp->use_auth)
    {
      if (serv->key_type == NIS_PK_DH)
	{
	  char netname[MAXNETNAMELEN + 1];
	  char *p;
	  des_block ckey;

	  p = stpcpy (netname, "unix@");
	  strncpy (p, serv->name, MAXNETNAMELEN - 5);
	  netname[MAXNETNAMELEN] = '\0';
	  dbp->clnt->cl_auth = NULL;
	  if (get_ckey (&ckey, &dbp->addr,
			dbp->use_udp ? IPPROTO_UDP : IPPROTO_TCP))
	    dbp->clnt->cl_auth =
	      authdes_pk_create (netname, &serv->pkey, 300, NULL, &ckey);
	  if (!dbp->clnt->cl_auth)
	    dbp->clnt->cl_auth = authunix_create_default ();
	}
      else
	dbp->clnt->cl_auth = authunix_create_default ();
    }

  return NIS_SUCCESS;
}
libnsl_hidden_nolink_def (__nisbind_connect, GLIBC_2_1)

nis_error
__nisbind_create (dir_binding *dbp, const nis_server *serv_val,
		  unsigned int serv_len, unsigned int server_used,
		  unsigned int current_ep, unsigned int flags)
{
  dbp->clnt = NULL;

  dbp->server_len = serv_len;
  dbp->server_val = (nis_server *)serv_val;

  if (flags & USE_DGRAM)
    dbp->use_udp = TRUE;
  else
    dbp->use_udp = FALSE;

  if (flags & NO_AUTHINFO)
    dbp->use_auth = FALSE;
  else
    dbp->use_auth = TRUE;

  if (flags & MASTER_ONLY)
    dbp->master_only = TRUE;
  else
    dbp->master_only = FALSE;

  /* We try the first server */
  dbp->trys = 1;

  dbp->class = -1;
  if (server_used == ~0)
    {
      if (__nis_findfastest (dbp) < 1)
	return NIS_NAMEUNREACHABLE;
    }
  else
    {
      dbp->server_used = server_used;
      dbp->current_ep = current_ep;
    }

  return NIS_SUCCESS;
}
libnsl_hidden_nolink_def (__nisbind_create, GLIBC_2_1)

/* __nisbind_connect (dbp) must be run before calling this function !
   So we could use the same binding twice */
nis_error
__do_niscall3 (dir_binding *dbp, u_long prog, xdrproc_t xargs, caddr_t req,
	       xdrproc_t xres, caddr_t resp, unsigned int flags, nis_cb *cb)
{
  enum clnt_stat result;
  nis_error retcode;

  if (dbp == NULL)
    return NIS_NAMEUNREACHABLE;

  do
    {
    again:
      result = clnt_call (dbp->clnt, prog, xargs, req, xres, resp, RPCTIMEOUT);

      if (result != RPC_SUCCESS)
	retcode = NIS_RPCERROR;
      else
	{
	  switch (prog)
	    {
	    case NIS_IBLIST:
	      if ((((nis_result *)resp)->status == NIS_CBRESULTS)
		  && (cb != NULL))
		{
		  __nis_do_callback (dbp, &((nis_result *) resp)->cookie, cb);
		  break;
		}
	      /* Yes, the missing break is correct. If we doesn't have to
		 start a callback, look if we have to search another server */
	      /* Fall through.  */
	    case NIS_LOOKUP:
	    case NIS_ADD:
	    case NIS_MODIFY:
	    case NIS_REMOVE:
	    case NIS_IBADD:
	    case NIS_IBMODIFY:
	    case NIS_IBREMOVE:
	    case NIS_IBFIRST:
	    case NIS_IBNEXT:
	      if (((nis_result *)resp)->status == NIS_SYSTEMERROR
		  || ((nis_result *)resp)->status == NIS_NOSUCHNAME
		  || ((nis_result *)resp)->status == NIS_NOT_ME)
		{
		next_server:
		  if (__nisbind_next (dbp) == NIS_SUCCESS)
		    {
		      while (__nisbind_connect (dbp) != NIS_SUCCESS)
			{
			  if (__nisbind_next (dbp) != NIS_SUCCESS)
			      return NIS_SUCCESS;
			}
		    }
		  else
		    break; /* No more servers to search in */
		  goto again;
		}
	      break;
	    case NIS_FINDDIRECTORY:
	      if (((fd_result *)resp)->status == NIS_SYSTEMERROR
		  || ((fd_result *)resp)->status == NIS_NOSUCHNAME
		  || ((fd_result *)resp)->status == NIS_NOT_ME)
		goto next_server;
	      break;
	    case NIS_DUMPLOG: /* log_result */
	    case NIS_DUMP:
	      if (((log_result *)resp)->lr_status == NIS_SYSTEMERROR
		  || ((log_result *)resp)->lr_status == NIS_NOSUCHNAME
		  || ((log_result *)resp)->lr_status == NIS_NOT_ME)
		goto next_server;
	      break;
	    default:
	      break;
	    }
	  retcode = NIS_SUCCESS;
	}
    }
  while ((flags & HARD_LOOKUP) && retcode == NIS_RPCERROR);

  return retcode;
}
libnsl_hidden_nolink_def (__do_niscall3, GLIBC_PRIVATE)


nis_error
__do_niscall2 (const nis_server *server, u_int server_len, u_long prog,
	       xdrproc_t xargs, caddr_t req, xdrproc_t xres, caddr_t resp,
	       unsigned int flags, nis_cb *cb)
{
  dir_binding dbp;
  nis_error status;

  if (flags & MASTER_ONLY)
    server_len = 1;

  status = __nisbind_create (&dbp, server, server_len, ~0, ~0, flags);
  if (status != NIS_SUCCESS)
    return status;

  while (__nisbind_connect (&dbp) != NIS_SUCCESS)
    if (__nisbind_next (&dbp) != NIS_SUCCESS)
      return NIS_NAMEUNREACHABLE;

  status = __do_niscall3 (&dbp, prog, xargs, req, xres, resp, flags, cb);

  __nisbind_destroy (&dbp);

  return status;

}

static directory_obj *
rec_dirsearch (const_nis_name name, directory_obj *dir, nis_error *status)
{
  fd_result *fd_res;
  XDR xdrs;

  switch (nis_dir_cmp (name, dir->do_name))
    {
    case SAME_NAME:
      *status = NIS_SUCCESS;
      return dir;
    case NOT_SEQUENTIAL:
      /* NOT_SEQUENTIAL means, go one up and try it there ! */
    case HIGHER_NAME:
      { /* We need data from a parent domain */
	directory_obj *obj;
	const char *ndomain = __nis_domain_of (dir->do_name);

	/* The root server of our domain is a replica of the parent
	   domain ! (Now I understand why a root server must be a
	   replica of the parent domain) */
	fd_res = __nis_finddirectory (dir, ndomain);
	if (fd_res == NULL)
	  {
	    nis_free_directory (dir);
	    *status = NIS_NOMEMORY;
	    return NULL;
	  }
	*status = fd_res->status;
	if (fd_res->status != NIS_SUCCESS)
	  {
	    /* Try the current directory obj, maybe it works */
	    __free_fdresult (fd_res);
	    return dir;
	  }
	nis_free_directory (dir);
	obj = calloc (1, sizeof (directory_obj));
	if (obj == NULL)
	  {
	    __free_fdresult (fd_res);
	    *status = NIS_NOMEMORY;
	    return NULL;
	  }
	xdrmem_create (&xdrs, fd_res->dir_data.dir_data_val,
		       fd_res->dir_data.dir_data_len, XDR_DECODE);
	_xdr_directory_obj (&xdrs, obj);
	xdr_destroy (&xdrs);
	__free_fdresult (fd_res);

	/* We have found a NIS+ server serving ndomain, now
	   let us search for "name" */
	return rec_dirsearch (name, obj, status);
      }
      break;
    case LOWER_NAME:
      {
	directory_obj *obj;
	size_t namelen = strlen (name);
	char leaf[namelen + 3];
	char domain[namelen + 3];
	const char *ndomain;
	char *cp;

	strcpy (domain, name);

	do
	  {
	    if (domain[0] == '\0')
	      {
		nis_free_directory (dir);
		return NULL;
	      }
	    nis_leaf_of_r (domain, leaf, sizeof (leaf));
	    ndomain = __nis_domain_of (domain);
	    memmove (domain, ndomain, strlen (ndomain) + 1);
	  }
	while (nis_dir_cmp (domain, dir->do_name) != SAME_NAME);

	cp = rawmemchr (leaf, '\0');
	*cp++ = '.';
	strcpy (cp, domain);

	fd_res = __nis_finddirectory (dir, leaf);
	if (fd_res == NULL)
	  {
	    nis_free_directory (dir);
	    *status = NIS_NOMEMORY;
	    return NULL;
	  }
	*status = fd_res->status;
	if (fd_res->status != NIS_SUCCESS)
	  {
	    /* Try the current directory object, maybe it works */
	    __free_fdresult (fd_res);
	    return dir;
	  }
	nis_free_directory (dir);
	obj = calloc (1, sizeof (directory_obj));
	if (obj == NULL)
	  {
	    __free_fdresult (fd_res);
	    *status = NIS_NOMEMORY;
	    return NULL;
	  }
	xdrmem_create (&xdrs, fd_res->dir_data.dir_data_val,
		       fd_res->dir_data.dir_data_len, XDR_DECODE);
	_xdr_directory_obj (&xdrs, obj);
	xdr_destroy (&xdrs);
	__free_fdresult (fd_res);
	/* We have found a NIS+ server serving ndomain, now
	   let us search for "name" */
	return rec_dirsearch (name, obj, status);
      }
      break;
    case BAD_NAME:
      nis_free_directory (dir);
      *status = NIS_BADNAME;
      return NULL;
    }
  nis_free_directory (dir);
  *status = NIS_FAIL;
  return NULL;
}

/* We try to query the current server for the searched object,
   maybe he know about it ? */
static directory_obj *
first_shoot (const_nis_name name, directory_obj *dir)
{
  directory_obj *obj = NULL;
  fd_result *fd_res;
  XDR xdrs;

  if (nis_dir_cmp (name, dir->do_name) == SAME_NAME)
    return dir;

  fd_res = __nis_finddirectory (dir, name);
  if (fd_res == NULL)
    return NULL;
  if (fd_res->status == NIS_SUCCESS
      && (obj = calloc (1, sizeof (directory_obj))) != NULL)
    {
      xdrmem_create (&xdrs, fd_res->dir_data.dir_data_val,
		     fd_res->dir_data.dir_data_len, XDR_DECODE);
      _xdr_directory_obj (&xdrs, obj);
      xdr_destroy (&xdrs);

      if (strcmp (dir->do_name, obj->do_name) != 0)
	{
	  nis_free_directory (obj);
	  obj = NULL;
	}
    }

  __free_fdresult (fd_res);

  if (obj != NULL)
    nis_free_directory (dir);

  return obj;
}

static struct nis_server_cache
{
  int search_parent;
  int uses;
  unsigned int size;
  unsigned int server_used;
  unsigned int current_ep;
  __time64_t expires;
  char name[];
} *nis_server_cache[16];
static time_t nis_cold_start_mtime;
__libc_lock_define_initialized (static, nis_server_cache_lock)

static directory_obj *
nis_server_cache_search (const_nis_name name, int search_parent,
			 unsigned int *server_used, unsigned int *current_ep,
			 struct __timespec64 *now)
{
  directory_obj *ret = NULL;
  int i;
  char *addr;
  XDR xdrs;
  struct stat64 st;

  int saved_errno = errno;
  if (stat64 ("/var/nis/NIS_COLD_START", &st) < 0)
    st.st_mtime = nis_cold_start_mtime + 1;
  __set_errno (saved_errno);

  __libc_lock_lock (nis_server_cache_lock);

  for (i = 0; i < 16; ++i)
    if (nis_server_cache[i] == NULL)
      continue;
    else if (st.st_mtime != nis_cold_start_mtime
	     || now->tv_sec > nis_server_cache[i]->expires)
      {
	free (nis_server_cache[i]);
	nis_server_cache[i] = NULL;
      }
    else if (nis_server_cache[i]->search_parent == search_parent
	     && strcmp (nis_server_cache[i]->name, name) == 0)
      {
	ret = calloc (1, sizeof (directory_obj));
	if (ret == NULL)
	  break;

	addr = rawmemchr (nis_server_cache[i]->name, '\0') + 8;
	addr = (char *) ((uintptr_t) addr & ~(uintptr_t) 7);
	xdrmem_create (&xdrs, addr, nis_server_cache[i]->size, XDR_DECODE);
	if (!_xdr_directory_obj (&xdrs, ret))
	  {
	    xdr_destroy (&xdrs);
	    free (ret);
	    ret = NULL;
	    free (nis_server_cache[i]);
	    nis_server_cache[i] = NULL;
	    break;
	  }
	xdr_destroy (&xdrs);
	*server_used = nis_server_cache[i]->server_used;
	*current_ep = nis_server_cache[i]->current_ep;
	break;
      }

  nis_cold_start_mtime = st.st_mtime;

  __libc_lock_unlock (nis_server_cache_lock);
  return ret;
}

static void
nis_server_cache_add (const_nis_name name, int search_parent,
		      directory_obj *dir, unsigned int server_used,
		      unsigned int current_ep, struct __timespec64 *now)
{
  struct nis_server_cache **loc;
  struct nis_server_cache *new;
  struct nis_server_cache *old;
  int i;
  char *addr;
  unsigned int size;
  XDR xdrs;

  if (dir == NULL)
    return;

  size = xdr_sizeof ((xdrproc_t) _xdr_directory_obj, (char *) dir);
  new = calloc (1, sizeof (*new) + strlen (name) + 8 + size);
  if (new == NULL)
    return;
  new->search_parent = search_parent;
  new->uses = 1;
  new->expires = now->tv_sec + dir->do_ttl;
  new->size = size;
  new->server_used = server_used;
  new->current_ep = current_ep;
  addr = stpcpy (new->name, name) + 8;
  addr = (char *) ((uintptr_t) addr & ~(uintptr_t) 7);

  xdrmem_create(&xdrs, addr, size, XDR_ENCODE);
  if (!_xdr_directory_obj (&xdrs, dir))
    {
      xdr_destroy (&xdrs);
      free (new);
      return;
    }
  xdr_destroy (&xdrs);

  __libc_lock_lock (nis_server_cache_lock);

  /* Choose which entry should be evicted from the cache.  */
  loc = &nis_server_cache[0];
  if (*loc != NULL)
    {
      for (i = 1; i < 16; ++i)
	if (nis_server_cache[i] == NULL)
	  {
	    loc = &nis_server_cache[i];
	    break;
	  }
	else if ((*loc)->uses > nis_server_cache[i]->uses
		 || ((*loc)->uses == nis_server_cache[i]->uses
		     && (*loc)->expires > nis_server_cache[i]->expires))
	  loc = &nis_server_cache[i];
    }
  old = *loc;
  *loc = new;

  __libc_lock_unlock (nis_server_cache_lock);
  free (old);
}

nis_error
__nisfind_server (const_nis_name name, int search_parent,
		  directory_obj **dir, dir_binding *dbp, unsigned int flags)
{
  nis_error result = NIS_SUCCESS;
  nis_error status;
  directory_obj *obj;
  struct __timespec64 ts;
  unsigned int server_used = ~0;
  unsigned int current_ep = ~0;

  if (name == NULL)
    return NIS_BADNAME;

  if (*dir != NULL)
    return NIS_SUCCESS;

  __clock_gettime64 (CLOCK_REALTIME, &ts);

  if ((flags & NO_CACHE) == 0)
    *dir = nis_server_cache_search (name, search_parent, &server_used,
				    &current_ep, &ts);
  if (*dir != NULL)
    {
      unsigned int server_len = (*dir)->do_servers.do_servers_len;
      if (flags & MASTER_ONLY)
	{
	  server_len = 1;
	  if (server_used != 0)
	    {
	      server_used = ~0;
	      current_ep = ~0;
	    }
	}
      result = __nisbind_create (dbp, (*dir)->do_servers.do_servers_val,
				 server_len, server_used, current_ep, flags);
      if (result != NIS_SUCCESS)
	{
	  nis_free_directory (*dir);
	  *dir = NULL;
	}
      return result;
    }

  int saved_errno = errno;
  *dir = readColdStartFile ();
  __set_errno (saved_errno);
  if (*dir == NULL)
    /* No /var/nis/NIS_COLD_START->no NIS+ installed.  */
    return NIS_UNAVAIL;

  /* Try at first, if servers in "dir" know our object */
  const char *search_name = name;
  if (search_parent)
    search_name = __nis_domain_of (name);
  obj = first_shoot (search_name, *dir);
  if (obj == NULL)
    {
      obj = rec_dirsearch (search_name, *dir, &status);
      if (obj == NULL)
	result = status;
    }

  if (result == NIS_SUCCESS)
    {
      unsigned int server_len = obj->do_servers.do_servers_len;
      if (flags & MASTER_ONLY)
	server_len = 1;
      result = __nisbind_create (dbp, obj->do_servers.do_servers_val,
				 server_len, ~0, ~0, flags);
      if (result == NIS_SUCCESS)
	{
	  if ((flags & MASTER_ONLY) == 0
	      || obj->do_servers.do_servers_len == 1)
	    {
	      server_used = dbp->server_used;
	      current_ep = dbp->current_ep;
	    }
	  if ((flags & NO_CACHE) == 0)
	    nis_server_cache_add (name, search_parent, obj,
				  server_used, current_ep, &ts);
	}
      else
	{
	  nis_free_directory (obj);
	  obj = NULL;
	}
    }

  *dir = obj;

  return result;
}


nis_error
__prepare_niscall (const_nis_name name, directory_obj **dirp,
		   dir_binding *bptrp, unsigned int flags)
{
  nis_error retcode = __nisfind_server (name, 1, dirp, bptrp, flags);
  if (__glibc_unlikely (retcode != NIS_SUCCESS))
    return retcode;

  do
    if (__nisbind_connect (bptrp) == NIS_SUCCESS)
      return NIS_SUCCESS;
  while (__nisbind_next (bptrp) == NIS_SUCCESS);

  __nisbind_destroy (bptrp);
  memset (bptrp, '\0', sizeof (*bptrp));

  retcode = NIS_NAMEUNREACHABLE;
  nis_free_directory (*dirp);
  *dirp = NULL;

  return retcode;
}
libnsl_hidden_nolink_def (__prepare_niscall, GLIBC_PRIVATE)


nis_error
__do_niscall (const_nis_name name, u_long prog, xdrproc_t xargs,
	      caddr_t req, xdrproc_t xres, caddr_t resp, unsigned int flags,
	      nis_cb *cb)
{
  dir_binding bptr;
  directory_obj *dir = NULL;
  int saved_errno = errno;

  nis_error retcode = __prepare_niscall (name, &dir, &bptr, flags);
  if (retcode == NIS_SUCCESS)
    {
      retcode = __do_niscall3 (&bptr, prog, xargs, req, xres, resp, flags, cb);

      __nisbind_destroy (&bptr);

      nis_free_directory (dir);
    }

  __set_errno (saved_errno);

  return retcode;
}
