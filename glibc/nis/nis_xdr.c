/* Copyright (c) 1997-2021 Free Software Foundation, Inc.
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

#include <stdint.h>
#include <rpcsvc/nis.h>
#include <rpcsvc/nis_callback.h> /* for "official" Solaris xdr functions */
#include <shlib-compat.h>

/* This functions do exist without beginning "_" under Solaris 2.x, but
   we have no prototypes for them. To avoid the same problems as with the
   YP xdr functions, we don't make them public. */
#include "nis_xdr.h"

static bool_t
xdr_nis_attr (XDR *xdrs, nis_attr *objp)
{
  bool_t res = xdr_string (xdrs, &objp->zattr_ndx, ~0);
  if (__builtin_expect (res, TRUE))
    res = xdr_bytes (xdrs, (char **) &objp->zattr_val.zattr_val_val,
		     &objp->zattr_val.zattr_val_len, ~0);
  return res;
}

static __always_inline bool_t
xdr_nis_name (XDR *xdrs, nis_name *objp)
{
  return xdr_string (xdrs, objp, ~0);
}

bool_t
_xdr_nis_name (XDR *xdrs, nis_name *objp)
{
  return xdr_nis_name (xdrs, objp);
}

static __always_inline bool_t
xdr_zotypes (XDR *xdrs, zotypes *objp)
{
  return xdr_enum (xdrs, (enum_t *) objp);
}

static __always_inline bool_t
xdr_nstype (XDR *xdrs, nstype *objp)
{
  return xdr_enum (xdrs, (enum_t *) objp);
}

static bool_t
xdr_oar_mask (XDR *xdrs, oar_mask *objp)
{
  bool_t res = xdr_u_int (xdrs, &objp->oa_rights);
  if (__builtin_expect (res, TRUE))
    res = xdr_zotypes (xdrs, &objp->oa_otype);
  return res;
}

static bool_t
xdr_endpoint (XDR *xdrs, endpoint *objp)
{
  bool_t res =  xdr_string (xdrs, &objp->uaddr, ~0);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_string (xdrs, &objp->family, ~0);
      if (__glibc_likely (res))
	res = xdr_string (xdrs, &objp->proto, ~0);
    }
  return res;
}

bool_t
_xdr_nis_server (XDR *xdrs, nis_server *objp)
{
  bool_t res = xdr_nis_name (xdrs, &objp->name);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_array (xdrs, (void *) &objp->ep.ep_val, &objp->ep.ep_len,
		       ~0, sizeof (endpoint), (xdrproc_t) xdr_endpoint);
      if (__builtin_expect (res, TRUE))
	{
	  res = xdr_u_int (xdrs, &objp->key_type);
	  if (__builtin_expect (res, TRUE))
	    res = xdr_netobj (xdrs, &objp->pkey);
	}
    }
  return res;
}

bool_t
_xdr_directory_obj (XDR *xdrs, directory_obj *objp)
{
  bool_t res = xdr_nis_name (xdrs, &objp->do_name);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_nstype (xdrs, &objp->do_type);
      if (__builtin_expect (res, TRUE))
	{
	  res = xdr_array (xdrs, (void *) &objp->do_servers.do_servers_val,
			   &objp->do_servers.do_servers_len, ~0,
			   sizeof (nis_server), (xdrproc_t) _xdr_nis_server);
	  if (__builtin_expect (res, TRUE))
	    {
	      res = xdr_uint32_t (xdrs, &objp->do_ttl);
	      if (__builtin_expect (res, TRUE))
		res = xdr_array (xdrs,
				 (void *) &objp->do_armask.do_armask_val,
				 &objp->do_armask.do_armask_len, ~0,
				 sizeof (oar_mask), (xdrproc_t) xdr_oar_mask);
	    }
	}
    }
  return res;
}

static bool_t
xdr_entry_col (XDR *xdrs, entry_col *objp)
{
  bool_t res = xdr_u_int (xdrs, &objp->ec_flags);
  if (__builtin_expect (res, TRUE))
    res = xdr_bytes (xdrs, (char **) &objp->ec_value.ec_value_val,
		     &objp->ec_value.ec_value_len, ~0);
  return res;
}

static bool_t
xdr_entry_obj (XDR *xdrs, entry_obj *objp)
{
  bool_t res = xdr_string (xdrs, &objp->en_type, ~0);
  if (__builtin_expect (res, TRUE))
    res = xdr_array (xdrs, (void *) &objp->en_cols.en_cols_val,
		     &objp->en_cols.en_cols_len, ~0,
		     sizeof (entry_col), (xdrproc_t) xdr_entry_col);
  return res;
}

static bool_t
xdr_group_obj (XDR *xdrs, group_obj *objp)
{
  bool_t res = xdr_u_int (xdrs, &objp->gr_flags);
  if (__builtin_expect (res, TRUE))
    res = xdr_array (xdrs, (void *) &objp->gr_members.gr_members_val,
		     &objp->gr_members.gr_members_len, ~0,
		     sizeof (nis_name), (xdrproc_t) _xdr_nis_name);
  return res;
}

static bool_t
xdr_link_obj (XDR *xdrs, link_obj *objp)
{
  bool_t res = xdr_zotypes (xdrs, &objp->li_rtype);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_array (xdrs, (void *) &objp->li_attrs.li_attrs_val,
		       &objp->li_attrs.li_attrs_len, ~0,
		       sizeof (nis_attr), (xdrproc_t) xdr_nis_attr);
      if (__builtin_expect (res, TRUE))
	res = xdr_nis_name (xdrs, &objp->li_name);
    }
  return res;
}

static bool_t
xdr_table_col (XDR *xdrs, table_col *objp)
{
  bool_t res = xdr_string (xdrs, &objp->tc_name, 64);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_u_int (xdrs, &objp->tc_flags);
      if (__builtin_expect (res, TRUE))
	res = xdr_u_int (xdrs, &objp->tc_rights);
    }
  return res;
}

static bool_t
xdr_table_obj (XDR *xdrs, table_obj *objp)
{
  bool_t res = xdr_string (xdrs, &objp->ta_type, 64);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_int (xdrs, &objp->ta_maxcol);
      if (__builtin_expect (res, TRUE))
	{
	  res = xdr_u_char (xdrs, &objp->ta_sep);
	  if (__builtin_expect (res, TRUE))
	    {
	      res = xdr_array (xdrs, (void *) &objp->ta_cols.ta_cols_val,
			       &objp->ta_cols.ta_cols_len, ~0,
			       sizeof (table_col), (xdrproc_t) xdr_table_col);
	      if (__builtin_expect (res, TRUE))
		res = xdr_string (xdrs, &objp->ta_path, ~0);
	    }
	}
    }
  return res;
}

static bool_t
xdr_objdata (XDR *xdrs, objdata *objp)
{
  bool_t res = xdr_zotypes (xdrs, &objp->zo_type);
  if (!__builtin_expect (res, TRUE))
    return res;
  switch (objp->zo_type)
    {
    case NIS_DIRECTORY_OBJ:
      return _xdr_directory_obj (xdrs, &objp->objdata_u.di_data);
    case NIS_GROUP_OBJ:
      return xdr_group_obj (xdrs, &objp->objdata_u.gr_data);
    case NIS_TABLE_OBJ:
      return xdr_table_obj (xdrs, &objp->objdata_u.ta_data);
    case NIS_ENTRY_OBJ:
      return xdr_entry_obj (xdrs, &objp->objdata_u.en_data);
    case NIS_LINK_OBJ:
      return xdr_link_obj (xdrs, &objp->objdata_u.li_data);
    case NIS_PRIVATE_OBJ:
      return xdr_bytes (xdrs, &objp->objdata_u.po_data.po_data_val,
			&objp->objdata_u.po_data.po_data_len, ~0);
    case NIS_NO_OBJ:
    case NIS_BOGUS_OBJ:
    default:
      return TRUE;
    }
}

static bool_t
xdr_nis_oid (XDR *xdrs, nis_oid *objp)
{
  bool_t res = xdr_uint32_t (xdrs, &objp->ctime);
  if  (__builtin_expect (res, TRUE))
    res = xdr_uint32_t (xdrs, &objp->mtime);
  return res;
}

bool_t
_xdr_nis_object (XDR *xdrs, nis_object *objp)
{
  bool_t res = xdr_nis_oid (xdrs, &objp->zo_oid);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_nis_name (xdrs, &objp->zo_name);
      if (__builtin_expect (res, TRUE))
	{
	  res = xdr_nis_name (xdrs, &objp->zo_owner);
	  if (__builtin_expect (res, TRUE))
	    {
	      res = xdr_nis_name (xdrs, &objp->zo_group);
	      if (__builtin_expect (res, TRUE))
		{
		  res = xdr_nis_name (xdrs, &objp->zo_domain);
		  if (__builtin_expect (res, TRUE))
		    {
		      res = xdr_u_int (xdrs, &objp->zo_access);
		      if (__builtin_expect (res, TRUE))
			{
			  res = xdr_uint32_t (xdrs, &objp->zo_ttl);
			  if (__builtin_expect (res, TRUE))
			    res = xdr_objdata (xdrs, &objp->zo_data);
			}
		    }
		}
	    }
	}
    }
  return res;
}

static __always_inline bool_t
xdr_nis_error (XDR *xdrs, nis_error *objp)
{
  return xdr_enum (xdrs, (enum_t *) objp);
}

bool_t
_xdr_nis_error (XDR *xdrs, nis_error *objp)
{
  return xdr_nis_error (xdrs, objp);
}

bool_t
_xdr_nis_result (XDR *xdrs, nis_result *objp)
{
  bool_t res = xdr_nis_error (xdrs, &objp->status);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_array (xdrs, (void *) &objp->objects.objects_val,
		       &objp->objects.objects_len, ~0,
		       sizeof (nis_object), (xdrproc_t) _xdr_nis_object);
      if (__builtin_expect (res, TRUE))
	{
	  res = xdr_netobj (xdrs, &objp->cookie);
	  if (__builtin_expect (res, TRUE))
	    {
	      res = xdr_uint32_t (xdrs, &objp->zticks);
	      if (__builtin_expect (res, TRUE))
		{
		  res = xdr_uint32_t (xdrs, &objp->dticks);
		  if (__builtin_expect (res, TRUE))
		    {
		      res = xdr_uint32_t (xdrs, &objp->aticks);
		      if (__builtin_expect (res, TRUE))
			res = xdr_uint32_t (xdrs, &objp->cticks);
		    }
		}
	    }
	}
    }
  return res;
}
libnsl_hidden_nolink_def (_xdr_nis_result, GLIBC_PRIVATE)

bool_t
_xdr_ns_request (XDR *xdrs, ns_request *objp)
{
  bool_t res = xdr_nis_name (xdrs, &objp->ns_name);
  if (__builtin_expect (res, TRUE))
    res = xdr_array (xdrs, (void *) &objp->ns_object.ns_object_val,
		     &objp->ns_object.ns_object_len, 1,
		     sizeof (nis_object), (xdrproc_t) _xdr_nis_object);
  return res;
}

bool_t
_xdr_ib_request (XDR *xdrs, ib_request *objp)
{
  bool_t res = xdr_nis_name (xdrs, &objp->ibr_name);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_array (xdrs, (void *) &objp->ibr_srch.ibr_srch_val,
		       &objp->ibr_srch.ibr_srch_len, ~0,
		       sizeof (nis_attr), (xdrproc_t) xdr_nis_attr);
      if (__builtin_expect (res, TRUE))
	{
	  res = xdr_u_int (xdrs, &objp->ibr_flags);
	  if (__builtin_expect (res, TRUE))
	    {
	      res = xdr_array (xdrs, (void *) &objp->ibr_obj.ibr_obj_val,
			       &objp->ibr_obj.ibr_obj_len, 1,
			       sizeof (nis_object),
			       (xdrproc_t) _xdr_nis_object);
	      if (__builtin_expect (res, TRUE))
		{
		  res = xdr_array (xdrs,
				   (void *) &objp->ibr_cbhost.ibr_cbhost_val,
				   &objp->ibr_cbhost.ibr_cbhost_len, 1,
				   sizeof (nis_server),
				   (xdrproc_t) _xdr_nis_server);
		  if (__builtin_expect (res, TRUE))
		    {
		      res = xdr_u_int (xdrs, &objp->ibr_bufsize);
		      if (__builtin_expect (res, TRUE))
			res =  xdr_netobj (xdrs, &objp->ibr_cookie);
		    }
		}
	    }
	}
    }
  return res;
}
libnsl_hidden_nolink_def (_xdr_ib_request, GLIBC_PRIVATE)

bool_t
_xdr_ping_args (XDR *xdrs, ping_args *objp)
{
  bool_t res = xdr_nis_name (xdrs, &objp->dir);
  if (__builtin_expect (res, TRUE))
    res = xdr_uint32_t (xdrs, &objp->stamp);
  return res;
}

bool_t
_xdr_cp_result (XDR *xdrs, cp_result *objp)
{
  bool_t res = xdr_nis_error (xdrs, &objp->cp_status);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_uint32_t (xdrs, &objp->cp_zticks);
      if (__builtin_expect (res, TRUE))
	res = xdr_uint32_t (xdrs, &objp->cp_dticks);
    }
  return res;
}

bool_t
_xdr_nis_tag (XDR *xdrs, nis_tag *objp)
{
  bool_t res = xdr_u_int (xdrs, &objp->tag_type);
  if (__builtin_expect (res, TRUE))
    res = xdr_string (xdrs, &objp->tag_val, ~0);
  return res;
}

bool_t
_xdr_nis_taglist (XDR *xdrs, nis_taglist *objp)
{
  return xdr_array (xdrs, (void *) &objp->tags.tags_val,
		    &objp->tags.tags_len, ~0, sizeof (nis_tag),
		    (xdrproc_t) _xdr_nis_tag);
}

bool_t
_xdr_fd_args (XDR *xdrs, fd_args *objp)
{
  bool_t res = xdr_nis_name (xdrs, &objp->dir_name);
  if (__builtin_expect (res, TRUE))
    res = xdr_nis_name (xdrs, &objp->requester);
  return res;
}

bool_t
_xdr_fd_result (XDR *xdrs, fd_result *objp)
{
  bool_t res = xdr_nis_error (xdrs, &objp->status);
  if (__builtin_expect (res, TRUE))
    {
      res = xdr_nis_name (xdrs, &objp->source);
      if (__builtin_expect (res, TRUE))
	{
	  res = xdr_bytes (xdrs, (char **) &objp->dir_data.dir_data_val,
			   &objp->dir_data.dir_data_len, ~0);
	  if (__builtin_expect (res, TRUE))
	    res = xdr_bytes (xdrs, (char **) &objp->signature.signature_val,
			     &objp->signature.signature_len, ~0);
	}
    }
  return res;
}

/* The following functions have prototypes in nis_callback.h.  So
   we make them public */
bool_t
xdr_obj_p (XDR *xdrs, obj_p *objp)
{
  return xdr_pointer (xdrs, (char **)objp, sizeof (nis_object),
		      (xdrproc_t)_xdr_nis_object);
}
libnsl_hidden_nolink_def (xdr_obj_p, GLIBC_2_1)

bool_t
xdr_cback_data (XDR *xdrs, cback_data *objp)
{
  return xdr_array (xdrs, (void *) &objp->entries.entries_val,
		    &objp->entries.entries_len, ~0,
		    sizeof (obj_p), (xdrproc_t) xdr_obj_p);
}
libnsl_hidden_nolink_def (xdr_cback_data, GLIBC_2_1)
