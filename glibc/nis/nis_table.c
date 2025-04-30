/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <string.h>
#include <rpcsvc/nis.h>
#include <libc-diag.h>
#include <shlib-compat.h>

#include "nis_xdr.h"
#include "nis_intern.h"
#include "libnsl.h"


struct ib_request *
__create_ib_request (const_nis_name name, unsigned int flags)
{
  struct ib_request *ibreq = calloc (1, sizeof (struct ib_request));
  nis_attr *search_val = NULL;
  size_t search_len = 0;
  size_t size = 0;

  if (ibreq == NULL)
    return NULL;

  ibreq->ibr_flags = flags;

  char *cptr = strdupa (name);

  /* Not of "[key=value,key=value,...],foo.." format? */
  if (cptr[0] != '[')
    {
      ibreq->ibr_name = strdup (cptr);
      if (ibreq->ibr_name == NULL)
	{
	  free (ibreq);
	  return NULL;
	}
      return ibreq;
    }

  /* "[key=value,...],foo" format */
  ibreq->ibr_name = strchr (cptr, ']');
  if (ibreq->ibr_name == NULL || ibreq->ibr_name[1] != ',')
    {
      /* The object has not really been built yet so we use free.  */
      free (ibreq);
      return NULL;
    }

  /* Check if we have an entry of "[key=value,],bar".  If, remove the "," */
  if (ibreq->ibr_name[-1] == ',')
    ibreq->ibr_name[-1] = '\0';
  else
    ibreq->ibr_name[0] = '\0';
  ibreq->ibr_name += 2;
  ibreq->ibr_name = strdup (ibreq->ibr_name);
  if (ibreq->ibr_name == NULL)
    {
    free_null:
      while (search_len-- > 0)
	{
	  free (search_val[search_len].zattr_ndx);
	  free (search_val[search_len].zattr_val.zattr_val_val);
	}
      free (search_val);
      nis_free_request (ibreq);
      return NULL;
    }

  ++cptr; /* Remove "[" */

  while (cptr != NULL && cptr[0] != '\0')
    {
      char *key = cptr;
      char *val = strchr (cptr, '=');

      cptr = strchr (key, ',');
      if (cptr != NULL)
	*cptr++ = '\0';

      if (__glibc_unlikely (val == NULL))
	{
	  nis_free_request (ibreq);
	  return NULL;
	}
      *val++ = '\0';
      if (search_len + 1 >= size)
	{
	  size += 1;
	  nis_attr *newp = realloc (search_val, size * sizeof (nis_attr));
	  if (newp == NULL)
	    goto free_null;
	  search_val = newp;
	}
      search_val[search_len].zattr_ndx = strdup (key);
      if (search_val[search_len].zattr_ndx == NULL)
	goto free_null;

      search_val[search_len].zattr_val.zattr_val_len = strlen (val) + 1;
      search_val[search_len].zattr_val.zattr_val_val = strdup (val);
      if (search_val[search_len].zattr_val.zattr_val_val == NULL)
	{
	  free (search_val[search_len].zattr_ndx);
	  goto free_null;
	}

      ++search_len;
    }

  ibreq->ibr_srch.ibr_srch_val = search_val;
  ibreq->ibr_srch.ibr_srch_len = search_len;

  return ibreq;
}
libnsl_hidden_nolink_def (__create_ib_request, GLIBC_PRIVATE)

static const struct timeval RPCTIMEOUT = {10, 0};

static char *
get_tablepath (char *name, dir_binding *bptr)
{
  enum clnt_stat result;
  nis_result res;
  struct ns_request req;

  memset (&res, '\0', sizeof (res));

  req.ns_name = name;
  req.ns_object.ns_object_len = 0;
  req.ns_object.ns_object_val = NULL;

  result = clnt_call (bptr->clnt, NIS_LOOKUP, (xdrproc_t) _xdr_ns_request,
		      (caddr_t) &req, (xdrproc_t) _xdr_nis_result,
		      (caddr_t) &res, RPCTIMEOUT);

  const char *cptr;
  if (result == RPC_SUCCESS && NIS_RES_STATUS (&res) == NIS_SUCCESS
      && __type_of (NIS_RES_OBJECT (&res)) == NIS_TABLE_OBJ)
    cptr = NIS_RES_OBJECT (&res)->TA_data.ta_path;
  else
    cptr = "";

  char *str = strdup (cptr);

  if (result == RPC_SUCCESS)
    xdr_free ((xdrproc_t) _xdr_nis_result, (char *) &res);

  return str;
}


nis_error
__follow_path (char **tablepath, char **tableptr, struct ib_request *ibreq,
	       dir_binding *bptr)
{
  if (*tablepath == NULL)
    {
      *tablepath = get_tablepath (ibreq->ibr_name, bptr);
      if (*tablepath == NULL)
	return NIS_NOMEMORY;

      *tableptr = *tablepath;
    }

  /* Since tableptr is only set here, and it's set when tablepath is NULL,
     which it is initially defined as, we know it will always be set here.  */
  DIAG_PUSH_NEEDS_COMMENT;
#if defined(__clang__)
  DIAG_IGNORE_NEEDS_COMMENT (4.7, "-Wsometimes-uninitialized");
#else
  DIAG_IGNORE_NEEDS_COMMENT (4.7, "-Wmaybe-uninitialized");
#endif

  if (*tableptr == NULL)
    return NIS_NOTFOUND;

  char *newname = strsep (tableptr, ":");
  if (newname[0] == '\0')
    return NIS_NOTFOUND;

  DIAG_POP_NEEDS_COMMENT;

  newname = strdup (newname);
  if (newname == NULL)
    return NIS_NOMEMORY;

  free (ibreq->ibr_name);
  ibreq->ibr_name = newname;

  return NIS_SUCCESS;
}
libnsl_hidden_nolink_def (__follow_path, GLIBC_PRIVATE)


nis_result *
nis_list (const_nis_name name, unsigned int flags,
	  int (*callback) (const_nis_name name,
			   const nis_object *object,
			   const void *userdata),
	  const void *userdata)
{
  nis_result *res = malloc (sizeof (nis_result));
  ib_request *ibreq;
  int status;
  enum clnt_stat clnt_status;
  int count_links = 0;		/* We will only follow NIS_MAXLINKS links! */
  int done = 0;
  nis_name *names;
  nis_name namebuf[2] = {NULL, NULL};
  int name_nr = 0;
  nis_cb *cb = NULL;
  char *tableptr;
  char *tablepath = NULL;
  int first_try = 0; /* Do we try the old binding at first ? */
  nis_result *allres = NULL;

  if (res == NULL)
    return NULL;

  if (name == NULL)
    {
      status = NIS_BADNAME;
    err_out:
      nis_freeresult (allres);
      memset (res, '\0', sizeof (nis_result));
      NIS_RES_STATUS (res) = status;
      return res;
    }

  ibreq = __create_ib_request (name, flags);
  if (ibreq == NULL)
    {
      status = NIS_BADNAME;
      goto err_out;
    }

  if ((flags & EXPAND_NAME)
      && ibreq->ibr_name[strlen (ibreq->ibr_name) - 1] != '.')
    {
      names = nis_getnames (ibreq->ibr_name);
      free (ibreq->ibr_name);
      ibreq->ibr_name = NULL;
      if (names == NULL)
	{
	  nis_free_request (ibreq);
	  status = NIS_BADNAME;
	  goto err_out;
	}
      ibreq->ibr_name = strdup (names[name_nr]);
      if (ibreq->ibr_name == NULL)
	{
	  nis_freenames (names);
	  nis_free_request (ibreq);
	  status = NIS_NOMEMORY;
	  goto err_out;
	}
    }
  else
    {
      names = namebuf;
      names[name_nr] = ibreq->ibr_name;
    }

  cb = NULL;

  while (!done)
    {
      dir_binding bptr;
      directory_obj *dir = NULL;

      memset (res, '\0', sizeof (nis_result));

      status = __nisfind_server (ibreq->ibr_name,
				 ibreq->ibr_srch.ibr_srch_val != NULL,
				 &dir, &bptr, flags & ~MASTER_ONLY);
      if (status != NIS_SUCCESS)
	{
	  NIS_RES_STATUS (res) = status;
	  goto fail3;
	}

      while (__nisbind_connect (&bptr) != NIS_SUCCESS)
	if (__glibc_unlikely (__nisbind_next (&bptr) != NIS_SUCCESS))
	  {
	    NIS_RES_STATUS (res) = NIS_NAMEUNREACHABLE;
	    goto fail;
	  }

      if (callback != NULL)
	{
	  assert (cb == NULL);
	  cb = __nis_create_callback (callback, userdata, flags);
	  ibreq->ibr_cbhost.ibr_cbhost_len = 1;
	  ibreq->ibr_cbhost.ibr_cbhost_val = cb->serv;
	}

    again:
      clnt_status = clnt_call (bptr.clnt, NIS_IBLIST,
			       (xdrproc_t) _xdr_ib_request, (caddr_t) ibreq,
			       (xdrproc_t) _xdr_nis_result,
			       (caddr_t) res, RPCTIMEOUT);

      if (__glibc_unlikely (clnt_status != RPC_SUCCESS))
	NIS_RES_STATUS (res) = NIS_RPCERROR;
      else
	switch (NIS_RES_STATUS (res))
	  { /* start switch */
	  case NIS_PARTIAL:
	  case NIS_SUCCESS:
	  case NIS_S_SUCCESS:
	    if (__type_of (NIS_RES_OBJECT (res)) == NIS_LINK_OBJ
		&& (flags & FOLLOW_LINKS))	/* We are following links.  */
	      {
		free (ibreq->ibr_name);
		ibreq->ibr_name = NULL;
		/* If we hit the link limit, bail.  */
		if (__glibc_unlikely (count_links > NIS_MAXLINKS))
		  {
		    NIS_RES_STATUS (res) = NIS_LINKNAMEERROR;
		    ++done;
		    break;
		  }
		++count_links;
		ibreq->ibr_name =
		  strdup (NIS_RES_OBJECT (res)->LI_data.li_name);
		if (ibreq->ibr_name == NULL)
		  {
		    NIS_RES_STATUS (res) = NIS_NOMEMORY;
		  fail:
		    __nisbind_destroy (&bptr);
		    nis_free_directory (dir);
		  fail3:
		    free (tablepath);
		    if (cb)
		      {
			__nis_destroy_callback (cb);
			ibreq->ibr_cbhost.ibr_cbhost_len = 0;
			ibreq->ibr_cbhost.ibr_cbhost_val = NULL;
		      }
		    if (names != namebuf)
		      nis_freenames (names);
		    nis_free_request (ibreq);
		    nis_freeresult (allres);
		    return res;
		  }
		if (NIS_RES_OBJECT (res)->LI_data.li_attrs.li_attrs_len)
		  if (ibreq->ibr_srch.ibr_srch_len == 0)
		    {
		      ibreq->ibr_srch.ibr_srch_len =
			NIS_RES_OBJECT (res)->LI_data.li_attrs.li_attrs_len;
		      ibreq->ibr_srch.ibr_srch_val =
			NIS_RES_OBJECT (res)->LI_data.li_attrs.li_attrs_val;
		    }
		/* The following is a non-obvious optimization.  A
		   nis_freeresult call would call xdr_free as the
		   following code.  But it also would unnecessarily
		   free the result structure.  We avoid this here
		   along with the necessary tests.  */
		xdr_free ((xdrproc_t) _xdr_nis_result, (char *)res);
		memset (res, '\0', sizeof (*res));
		first_try = 1; /* Try at first the old binding */
		goto again;
	      }
	    else if ((flags & FOLLOW_PATH)
		     && NIS_RES_STATUS (res) == NIS_PARTIAL)
	      {
		enum nis_error err = __follow_path (&tablepath, &tableptr,
						    ibreq, &bptr);
		if (err != NIS_SUCCESS)
		  {
		    if (err == NIS_NOMEMORY)
		      NIS_RES_STATUS (res) = err;
		    ++done;
		  }
		else
		  {
		    /* The following is a non-obvious optimization.  A
		       nis_freeresult call would call xdr_free as the
		       following code.  But it also would unnecessarily
		       free the result structure.  We avoid this here
		       along with the necessary tests.  */
		    xdr_free ((xdrproc_t) _xdr_nis_result, (char *) res);
		    memset (res, '\0', sizeof (*res));
		    first_try = 1;
		    goto again;
		  }
	      }
	    else if ((flags & (FOLLOW_PATH | ALL_RESULTS))
		     == (FOLLOW_PATH | ALL_RESULTS))
	      {
		if (allres == NULL)
		  {
		    allres = res;
		    res = malloc (sizeof (nis_result));
		    if (res == NULL)
		      {
			res = allres;
			allres = NULL;
			NIS_RES_STATUS (res) = NIS_NOMEMORY;
			goto fail;
		      }
		    NIS_RES_STATUS (res) = NIS_RES_STATUS (allres);
		  }
		else
		  {
		    nis_object *objects_val
		      = realloc (NIS_RES_OBJECT (allres),
				 (NIS_RES_NUMOBJ (allres)
				  + NIS_RES_NUMOBJ (res))
				 * sizeof (nis_object));
		    if (objects_val == NULL)
		      {
			NIS_RES_STATUS (res) = NIS_NOMEMORY;
			goto fail;
		      }
		    NIS_RES_OBJECT (allres) = objects_val;
		    memcpy (NIS_RES_OBJECT (allres) + NIS_RES_NUMOBJ (allres),
			    NIS_RES_OBJECT (res),
			    NIS_RES_NUMOBJ (res) * sizeof (nis_object));
		    NIS_RES_NUMOBJ (allres) += NIS_RES_NUMOBJ (res);
		    NIS_RES_NUMOBJ (res) = 0;
		    free (NIS_RES_OBJECT (res));
		    NIS_RES_OBJECT (res) = NULL;
		    NIS_RES_STATUS (allres) = NIS_RES_STATUS (res);
		    xdr_free ((xdrproc_t) _xdr_nis_result, (char *) res);
		  }
		enum nis_error err = __follow_path (&tablepath, &tableptr,
						    ibreq, &bptr);
		if (err != NIS_SUCCESS)
		  {
		    /* Prepare for the nis_freeresult call.  */
		    memset (res, '\0', sizeof (*res));

		    if (err == NIS_NOMEMORY)
		      NIS_RES_STATUS (allres) = err;
		    ++done;
		  }
	      }
	    else
	      ++done;
	    break;
	  case NIS_CBRESULTS:
	    if (cb != NULL)
	      {
		__nis_do_callback (&bptr, &res->cookie, cb);
		NIS_RES_STATUS (res) = cb->result;

		if (!(flags & ALL_RESULTS))
		  ++done;
		else
		  {
		    enum nis_error err
		      = __follow_path (&tablepath, &tableptr, ibreq, &bptr);
		    if (err != NIS_SUCCESS)
		      {
			if (err == NIS_NOMEMORY)
			  NIS_RES_STATUS (res) = err;
			++done;
		      }
		  }
	      }
	    break;
	  case NIS_SYSTEMERROR:
	  case NIS_NOSUCHNAME:
	  case NIS_NOT_ME:
	    /* If we had first tried the old binding, do nothing, but
	       get a new binding */
	    if (!first_try)
	      {
		if (__nisbind_next (&bptr) != NIS_SUCCESS)
		  {
		    ++done;
		    break; /* No more servers to search */
		  }
		while (__nisbind_connect (&bptr) != NIS_SUCCESS)
		  {
		    if (__nisbind_next (&bptr) != NIS_SUCCESS)
		      {
			++done;
			break; /* No more servers to search */
		      }
		  }
		goto again;
	      }
	    break;
	  default:
	    if (!first_try)
	      {
		/* Try the next domainname if we don't follow a link.  */
		free (ibreq->ibr_name);
		ibreq->ibr_name = NULL;
		if (__glibc_unlikely (count_links))
		  {
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
		ibreq->ibr_name = strdup (names[name_nr]);
		if (ibreq->ibr_name == NULL)
		  {
		    NIS_RES_STATUS (res) = NIS_NOMEMORY;
		    goto fail;
		  }
		first_try = 1; /* Try old binding at first */
		goto again;
	      }
	    break;
	  }
      first_try = 0;

      if (cb)
	{
	  __nis_destroy_callback (cb);
	  ibreq->ibr_cbhost.ibr_cbhost_len = 0;
	  ibreq->ibr_cbhost.ibr_cbhost_val = NULL;
	  cb = NULL;
	}

      __nisbind_destroy (&bptr);
      nis_free_directory (dir);
    }

  free (tablepath);

  if (names != namebuf)
    nis_freenames (names);

  nis_free_request (ibreq);

  if (allres)
    {
      nis_freeresult (res);
      return allres;
    }

  return res;
}
libnsl_hidden_nolink_def (nis_list, GLIBC_2_1)

nis_result *
nis_add_entry (const_nis_name name, const nis_object *obj2, unsigned int flags)
{
  nis_result *res = calloc (1, sizeof (nis_result));
  if (res == NULL)
    return NULL;

  if (name == NULL)
    {
      NIS_RES_STATUS (res) = NIS_BADNAME;
      return res;
    }

  ib_request *ibreq = __create_ib_request (name, flags);
  if (ibreq == NULL)
    {
      NIS_RES_STATUS (res) = NIS_BADNAME;
      return res;
    }

  nis_object obj;
  memcpy (&obj, obj2, sizeof (nis_object));

  size_t namelen = strlen (name);
  char buf1[namelen + 20];
  char buf4[namelen + 20];

  if (obj.zo_name == NULL || strlen (obj.zo_name) == 0)
    obj.zo_name = nis_leaf_of_r (name, buf1, sizeof (buf1));

  if (obj.zo_owner == NULL || strlen (obj.zo_owner) == 0)
    obj.zo_owner = nis_local_principal ();

  if (obj.zo_group == NULL || strlen (obj.zo_group) == 0)
    obj.zo_group = nis_local_group ();

  obj.zo_domain = nis_domain_of_r (name, buf4, sizeof (buf4));

  ibreq->ibr_obj.ibr_obj_val = nis_clone_object (&obj, NULL);
  if (ibreq->ibr_obj.ibr_obj_val == NULL)
    {
      nis_free_request (ibreq);
      NIS_RES_STATUS (res) = NIS_NOMEMORY;
      return res;
    }
  ibreq->ibr_obj.ibr_obj_len = 1;

  nis_error status = __do_niscall (ibreq->ibr_name, NIS_IBADD,
				   (xdrproc_t) _xdr_ib_request,
				   (caddr_t) ibreq,
				   (xdrproc_t) _xdr_nis_result,
				   (caddr_t) res, 0, NULL);
  if (__glibc_unlikely (status != NIS_SUCCESS))
    NIS_RES_STATUS (res) = status;

  nis_free_request (ibreq);

  return res;
}
libnsl_hidden_nolink_def (nis_add_entry, GLIBC_2_1)

nis_result *
nis_modify_entry (const_nis_name name, const nis_object *obj2,
		  unsigned int flags)
{
  nis_object obj;
  nis_result *res;
  nis_error status;
  ib_request *ibreq;
  size_t namelen = strlen (name);
  char buf1[namelen + 20];
  char buf4[namelen + 20];

  res = calloc (1, sizeof (nis_result));
  if (res == NULL)
    return NULL;

  ibreq = __create_ib_request (name, flags);
  if (ibreq == NULL)
    {
      NIS_RES_STATUS (res) = NIS_BADNAME;
      return res;
    }

  memcpy (&obj, obj2, sizeof (nis_object));

  if (obj.zo_name == NULL || strlen (obj.zo_name) == 0)
    obj.zo_name = nis_leaf_of_r (name, buf1, sizeof (buf1));

  if (obj.zo_owner == NULL || strlen (obj.zo_owner) == 0)
    obj.zo_owner = nis_local_principal ();

  if (obj.zo_group == NULL || strlen (obj.zo_group) == 0)
    obj.zo_group = nis_local_group ();

  obj.zo_domain = nis_domain_of_r (name, buf4, sizeof (buf4));

  ibreq->ibr_obj.ibr_obj_val = nis_clone_object (&obj, NULL);
  if (ibreq->ibr_obj.ibr_obj_val == NULL)
    {
      nis_free_request (ibreq);
      NIS_RES_STATUS (res) = NIS_NOMEMORY;
      return res;
    }
  ibreq->ibr_obj.ibr_obj_len = 1;

  status = __do_niscall (ibreq->ibr_name, NIS_IBMODIFY,
			 (xdrproc_t) _xdr_ib_request,
			 (caddr_t) ibreq, (xdrproc_t) _xdr_nis_result,
			 (caddr_t) res, 0, NULL);
  if (__glibc_unlikely (status != NIS_SUCCESS))
    NIS_RES_STATUS (res) = status;

  nis_free_request (ibreq);

  return res;
}
libnsl_hidden_nolink_def (nis_modify_entry, GLIBC_2_1)

nis_result *
nis_remove_entry (const_nis_name name, const nis_object *obj,
		  unsigned int flags)
{
  nis_result *res;
  ib_request *ibreq;
  nis_error status;

  res = calloc (1, sizeof (nis_result));
  if (res == NULL)
    return NULL;

  if (name == NULL)
    {
      NIS_RES_STATUS (res) = NIS_BADNAME;
      return res;
    }

  ibreq = __create_ib_request (name, flags);
  if (ibreq == NULL)
    {
      NIS_RES_STATUS (res) = NIS_BADNAME;
      return res;
    }

  if (obj != NULL)
    {
      ibreq->ibr_obj.ibr_obj_val = nis_clone_object (obj, NULL);
      if (ibreq->ibr_obj.ibr_obj_val == NULL)
	{
	  nis_free_request (ibreq);
	  NIS_RES_STATUS (res) = NIS_NOMEMORY;
	  return res;
	}
      ibreq->ibr_obj.ibr_obj_len = 1;
    }

  if ((status = __do_niscall (ibreq->ibr_name, NIS_IBREMOVE,
			      (xdrproc_t) _xdr_ib_request,
			      (caddr_t) ibreq, (xdrproc_t) _xdr_nis_result,
			      (caddr_t) res, 0, NULL)) != NIS_SUCCESS)
    NIS_RES_STATUS (res) = status;

  nis_free_request (ibreq);

  return res;
}
libnsl_hidden_nolink_def (nis_remove_entry, GLIBC_2_1)

nis_result *
nis_first_entry (const_nis_name name)
{
  nis_result *res;
  ib_request *ibreq;
  nis_error status;

  res = calloc (1, sizeof (nis_result));
  if (res == NULL)
    return NULL;

  if (name == NULL)
    {
      NIS_RES_STATUS (res) = NIS_BADNAME;
      return res;
    }

  ibreq = __create_ib_request (name, 0);
  if (ibreq == NULL)
    {
      NIS_RES_STATUS (res) = NIS_BADNAME;
      return res;
    }

  status = __do_niscall (ibreq->ibr_name, NIS_IBFIRST,
			 (xdrproc_t) _xdr_ib_request,
			 (caddr_t) ibreq, (xdrproc_t) _xdr_nis_result,
			 (caddr_t) res, 0, NULL);

  if (__glibc_unlikely (status != NIS_SUCCESS))
    NIS_RES_STATUS (res) = status;

  nis_free_request (ibreq);

  return res;
}
libnsl_hidden_nolink_def (nis_first_entry, GLIBC_2_1)

nis_result *
nis_next_entry (const_nis_name name, const netobj *cookie)
{
  nis_result *res;
  ib_request *ibreq;
  nis_error status;

  res = calloc (1, sizeof (nis_result));
  if (res == NULL)
    return NULL;

  if (name == NULL)
    {
      NIS_RES_STATUS (res) = NIS_BADNAME;
      return res;
    }

  ibreq = __create_ib_request (name, 0);
  if (ibreq == NULL)
    {
      NIS_RES_STATUS (res) = NIS_BADNAME;
      return res;
    }

  if (cookie != NULL)
    {
      ibreq->ibr_cookie.n_bytes = cookie->n_bytes;
      ibreq->ibr_cookie.n_len = cookie->n_len;
    }

  status = __do_niscall (ibreq->ibr_name, NIS_IBNEXT,
			 (xdrproc_t) _xdr_ib_request,
			 (caddr_t) ibreq, (xdrproc_t) _xdr_nis_result,
			 (caddr_t) res, 0, NULL);

  if (__glibc_unlikely (status != NIS_SUCCESS))
    NIS_RES_STATUS (res) = status;

  if (cookie != NULL)
    {
      /* Don't give cookie free, it is not from us */
      ibreq->ibr_cookie.n_bytes = NULL;
      ibreq->ibr_cookie.n_len = 0;
    }

  nis_free_request (ibreq);

  return res;
}
libnsl_hidden_nolink_def (nis_next_entry, GLIBC_2_1)
