/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <netdb.h>
#include "nsswitch.h"
#include <resolv/resolv_context.h>

/* Set up NIP to run through the services.  If ALL is zero, use NIP's
   current location if it's not nil.  Return nonzero if there are no
   services (left).  */
static int
setup (const char *func_name, db_lookup_function lookup_fct,
       void **fctp, nss_action_list *nip, nss_action_list *startp, int all)
{
  int no_more;
  if (*startp == NULL || all)
    {
      no_more = lookup_fct (nip, func_name, NULL, fctp);
      *startp = no_more ? (nss_action_list) -1l : *nip;
    }
  else if (*startp == (nss_action_list) -1l)
    /* No services at all.  */
    return 1;
  else
    {
      if (!*nip)
	/* Reset to the beginning of the service list.  */
	*nip = *startp;
      /* Look up the first function.  */
      no_more = __nss_lookup (nip, func_name, NULL, fctp);
    }
  return no_more;
}

void
__nss_setent (const char *func_name, db_lookup_function lookup_fct,
	      nss_action_list *nip, nss_action_list *startp,
	      nss_action_list *last_nip, int stayopen, int *stayopen_tmp,
	      int res)
{
  union
  {
    setent_function f;
    void *ptr;
  } fct;
  int no_more;

  struct resolv_context *res_ctx = NULL;
  if (res)
    {
      res_ctx = __resolv_context_get ();
      if (res_ctx == NULL)
	{
	  __set_h_errno (NETDB_INTERNAL);
	  return;
	}
    }

  /* Cycle through the services and run their `setXXent' functions until
     we find an available service.  */
  no_more = setup (func_name, lookup_fct, &fct.ptr, nip,
		   startp, 1);
  while (! no_more)
    {
      int is_last_nip = *nip == *last_nip;
      enum nss_status status;

      if (stayopen_tmp)
	status = DL_CALL_FCT (fct.f, (*stayopen_tmp));
      else
	status = DL_CALL_FCT (fct.f, (0));


      /* This is a special-case.  When [SUCCESS=merge] is in play,
         _nss_next2() will skip to the next database.  Due to the
         implementation of that function, we can't know whether we're
         in an enumeration or an individual lookup, which behaves
         differently with regards to merging.  We'll treat SUCCESS as
         an indication to start the enumeration at this database. */
      if (nss_next_action (*nip, status) == NSS_ACTION_MERGE)
	no_more = 1;
      else
	no_more = __nss_next2 (nip, func_name, NULL, &fct.ptr, status, 0);

      if (is_last_nip)
	*last_nip = *nip;
    }

  __resolv_context_put (res_ctx);

  if (stayopen_tmp)
    *stayopen_tmp = stayopen;
}


void
__nss_endent (const char *func_name, db_lookup_function lookup_fct,
	      nss_action_list *nip, nss_action_list *startp,
	      nss_action_list *last_nip, int res)
{
  union
  {
    endent_function f;
    void *ptr;
  } fct;
  int no_more;

  struct resolv_context *res_ctx = NULL;
  if (res)
    {
      res_ctx = __resolv_context_get ();
      if (res_ctx == NULL)
	{
	  __set_h_errno (NETDB_INTERNAL);
	  return;
	}
    }

  /* Cycle through all the services and run their endXXent functions.  */
  no_more = setup (func_name, lookup_fct, &fct.ptr, nip, startp, 1);
  while (! no_more)
    {
      /* Ignore status, we force check in __NSS_NEXT.  */
      DL_CALL_FCT (fct.f, ());

      if (*nip == *last_nip)
	/* We have processed all services which were used.  */
	break;

      no_more = __nss_next2 (nip, func_name, NULL, &fct.ptr, 0, 1);
    }
  *last_nip = *nip = NULL;

  __resolv_context_put (res_ctx);
}


int
__nss_getent_r (const char *getent_func_name,
		const char *setent_func_name,
		db_lookup_function lookup_fct,
		nss_action_list *nip, nss_action_list *startp,
		nss_action_list *last_nip, int *stayopen_tmp, int res,
		void *resbuf, char *buffer, size_t buflen,
		void **result, int *h_errnop)
{
  union
  {
    getent_function f;
    void *ptr;
  } fct;
  int no_more;
  enum nss_status status;

  struct resolv_context *res_ctx = NULL;
  if (res)
    {
      res_ctx = __resolv_context_get ();
      if (res_ctx == NULL)
	{
	  *h_errnop = NETDB_INTERNAL;
	  *result = NULL;
	  return errno;
	}
    }

  /* Initialize status to return if no more functions are found.  */
  status = NSS_STATUS_NOTFOUND;

  /* Run through available functions, starting with the same function last
     run.  We will repeat each function as long as it succeeds, and then go
     on to the next service action.  */
  no_more = setup (getent_func_name, lookup_fct, &fct.ptr, nip,
		   startp, 0);
  while (! no_more)
    {
      int is_last_nip = *nip == *last_nip;

      status = DL_CALL_FCT (fct.f,
			    (resbuf, buffer, buflen, &errno, &h_errno));

      /* The status is NSS_STATUS_TRYAGAIN and errno is ERANGE the
	 provided buffer is too small.  In this case we should give
	 the user the possibility to enlarge the buffer and we should
	 not simply go on with the next service (even if the TRYAGAIN
	 action tells us so).  */
      if (status == NSS_STATUS_TRYAGAIN
	  && (h_errnop == NULL || *h_errnop == NETDB_INTERNAL)
	  && errno == ERANGE)
	break;

      do
	{
        /* This is a special-case.  When [SUCCESS=merge] is in play,
           _nss_next2() will skip to the next database.  Due to the
           implementation of that function, we can't know whether we're
           in an enumeration or an individual lookup, which behaves
           differently with regards to merging.  We'll treat SUCCESS as
           an indication to return the results here. */
	  if (status == NSS_STATUS_SUCCESS
	      && nss_next_action (*nip, status) == NSS_ACTION_MERGE)
	    no_more = 1;
	  else
	    no_more = __nss_next2 (nip, getent_func_name, NULL, &fct.ptr,
				   status, 0);

	  if (is_last_nip)
	    *last_nip = *nip;

	  if (! no_more)
	    {
	      /* Call the `setXXent' function.  This wasn't done before.  */
	      union
	      {
		setent_function f;
		void *ptr;
	      } sfct;

	      no_more = __nss_lookup (nip, setent_func_name, NULL, &sfct.ptr);

	      if (! no_more)
	        {
		  if (stayopen_tmp)
		    status = DL_CALL_FCT (sfct.f, (*stayopen_tmp));
		  else
		    status = DL_CALL_FCT (sfct.f, (0));
		}
	      else
		status = NSS_STATUS_NOTFOUND;
	    }
	}
      while (! no_more && status != NSS_STATUS_SUCCESS);
    }

  __resolv_context_put (res_ctx);

  *result = status == NSS_STATUS_SUCCESS ? resbuf : NULL;
  return (status == NSS_STATUS_SUCCESS ? 0
	  : status != NSS_STATUS_TRYAGAIN ? ENOENT
	  /* h_errno functions only set errno if h_errno is NETDB_INTERNAL.  */
	  : (h_errnop == NULL || *h_errnop == NETDB_INTERNAL) ? errno
	  : EAGAIN);
}
