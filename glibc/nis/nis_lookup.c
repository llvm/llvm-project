/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@uni-paderborn.de>, 1997.

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

#include <string.h>
#include <rpcsvc/nis.h>
#include "nis_xdr.h"
#include "nis_intern.h"
#include <libnsl.h>
#include <shlib-compat.h>


nis_result *
nis_lookup (const_nis_name name, const unsigned int flags)
{
  nis_result *res = calloc (1, sizeof (nis_result));
  struct ns_request req;
  nis_name *names;
  nis_error status;
  int link_first_try = 0;
  int count_links = 0;	 /* We will follow only 16 links in the deep */
  int done = 0;
  int name_nr = 0;
  nis_name namebuf[2] = {NULL, NULL};

  if (res == NULL)
    return NULL;

  if ((flags & EXPAND_NAME) && (name[strlen (name) - 1] != '.'))
    {
      names = nis_getnames (name);
      if (names == NULL)
	{
	  NIS_RES_STATUS (res) = NIS_NAMEUNREACHABLE;
	  return res;
	}
    }
  else
    {
      names = namebuf;
      names[0] = (nis_name)name;
    }

  req.ns_name = names[0];
  while (!done)
    {
      dir_binding bptr;
      directory_obj *dir = NULL;
      req.ns_object.ns_object_len = 0;
      req.ns_object.ns_object_val = NULL;

      status = __prepare_niscall (req.ns_name, &dir, &bptr, flags);
      if (__glibc_unlikely (status != NIS_SUCCESS))
	{
	  NIS_RES_STATUS (res) = status;
	  goto out;
	}

      do
	{
	  static const struct timeval RPCTIMEOUT = {10, 0};
	  enum clnt_stat result;

	again:
	  result = clnt_call (bptr.clnt, NIS_LOOKUP,
			      (xdrproc_t) _xdr_ns_request,
			      (caddr_t) &req, (xdrproc_t) _xdr_nis_result,
			      (caddr_t) res, RPCTIMEOUT);

	  if (result != RPC_SUCCESS)
	    status = NIS_RPCERROR;
	  else
	    {
	      status = NIS_SUCCESS;

	      if (NIS_RES_STATUS (res) == NIS_SUCCESS)
		{
		    if (__type_of (NIS_RES_OBJECT (res)) == NIS_LINK_OBJ
			&& (flags & FOLLOW_LINKS)) /* We are following links */
		      {
			/* if we hit the link limit, bail */
			if (count_links > NIS_MAXLINKS)
			  {
			    NIS_RES_STATUS (res) = NIS_LINKNAMEERROR;
			    break;
			  }
			++count_links;
			req.ns_name =
			  strdupa (NIS_RES_OBJECT (res)->LI_data.li_name);

			/* The following is a non-obvious optimization.  A
			   nis_freeresult call would call xdr_free as the
			   following code.  But it also would unnecessarily
			   free the result structure.  We avoid this here
			   along with the necessary tests.  */
			xdr_free ((xdrproc_t) _xdr_nis_result, (char *) res);
			memset (res, '\0', sizeof (*res));

			link_first_try = 1; /* Try at first the old binding */
			goto again;
		      }
		}
	      else
		if (NIS_RES_STATUS (res) == NIS_SYSTEMERROR
		    || NIS_RES_STATUS (res) == NIS_NOSUCHNAME
		    || NIS_RES_STATUS (res) == NIS_NOT_ME)
		  {
		    if (link_first_try)
		      {
			__nisbind_destroy (&bptr);
			nis_free_directory (dir);
			/* Otherwise __nisfind_server will not do anything.  */
			dir = NULL;

			if (__nisfind_server (req.ns_name, 1, &dir, &bptr,
					      flags & ~MASTER_ONLY)
			    != NIS_SUCCESS)
			  goto out;
		      }
		    else
		      if (__nisbind_next (&bptr) != NIS_SUCCESS)
			{
			  /* No more servers to search.  Try parent.  */
			  const char *ndomain = __nis_domain_of (req.ns_name);
			  req.ns_name = strdupa (ndomain);
			  if (strcmp (req.ns_name, ".") == 0)
			    {
			      NIS_RES_STATUS (res) = NIS_NAMEUNREACHABLE;
			      goto out;
			    }

			  __nisbind_destroy (&bptr);
			  nis_free_directory (dir);
			  dir = NULL;
			  status = __prepare_niscall (req.ns_name, &dir,
						      &bptr, flags);
			  if (__glibc_unlikely (status != NIS_SUCCESS))
			    {
			      NIS_RES_STATUS (res) = status;
			      goto out;
			    }
			  goto again;
			}

		    while (__nisbind_connect (&bptr) != NIS_SUCCESS)
		      {
			if (__nisbind_next (&bptr) != NIS_SUCCESS)
			  {
			    nis_free_directory (dir);
			    goto out;
			  }
		      }
		    goto again;
		  }
	      break;
	    }
	  link_first_try = 0; /* Set it back */
	}
      while ((flags & HARD_LOOKUP) && status == NIS_RPCERROR);

      __nisbind_destroy (&bptr);
      nis_free_directory (dir);

      if (status != NIS_SUCCESS)
	{
	  NIS_RES_STATUS (res) = status;
	  goto out;
	}

      switch (NIS_RES_STATUS (res))
	{
	case NIS_PARTIAL:
	case NIS_SUCCESS:
	case NIS_S_SUCCESS:
	case NIS_LINKNAMEERROR: /* We follow to max links */
	case NIS_UNAVAIL: /* NIS+ is not installed, or all servers are down */
	  ++done;
	  break;
	default:
	  /* Try the next domainname if we don't follow a link */
	  if (count_links)
	    {
	      free (req.ns_name);
	      NIS_RES_STATUS (res) = NIS_LINKNAMEERROR;
	      ++done;
	      break;
	    }
	  ++name_nr;
	  if (names[name_nr] == NULL)
	    {
	      ++done;
	      break;
	    }
	  req.ns_name = names[name_nr];
	  break;
	}
    }

 out:
  if (names != namebuf)
    nis_freenames (names);

  return res;
}
libnsl_hidden_nolink_def (nis_lookup, GLIBC_2_1)
