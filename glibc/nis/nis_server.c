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

#include <string.h>
#include <rpcsvc/nis.h>
#include <shlib-compat.h>

#include "nis_xdr.h"
#include "nis_intern.h"

nis_error
nis_servstate (const nis_server *serv, const nis_tag *tags,
	       const int numtags, nis_tag **result)
{
  nis_taglist taglist;
  nis_taglist tagres;

  *result = 0;
  tagres.tags.tags_len = 0;
  tagres.tags.tags_val = NULL;
  taglist.tags.tags_len = numtags;
  taglist.tags.tags_val = (nis_tag *) tags;

  if (serv == NULL)
    return NIS_BADOBJECT;

  if (__do_niscall2 (serv, 1, NIS_SERVSTATE, (xdrproc_t) _xdr_nis_taglist,
		     (caddr_t) &taglist, (xdrproc_t) _xdr_nis_taglist,
		     (caddr_t) &tagres, 0, NULL) != NIS_SUCCESS)
    return NIS_RPCERROR;

  *result = tagres.tags.tags_val;

  return NIS_SUCCESS;
}
libnsl_hidden_nolink_def (nis_servstate, GLIBC_2_1)

nis_error
nis_stats (const nis_server *serv, const nis_tag *tags,
	   const int numtags, nis_tag **result)
{
  nis_taglist taglist;
  nis_taglist tagres;

  *result = NULL;
  tagres.tags.tags_len = 0;
  tagres.tags.tags_val = NULL;
  taglist.tags.tags_len = numtags;
  taglist.tags.tags_val = (nis_tag *) tags;

  if (serv == NULL)
    return NIS_BADOBJECT;

  if (__do_niscall2 (serv, 1, NIS_STATUS, (xdrproc_t) _xdr_nis_taglist,
		     (caddr_t) &taglist, (xdrproc_t) _xdr_nis_taglist,
		     (caddr_t) &tagres, 0, NULL) != NIS_SUCCESS)
    return NIS_RPCERROR;

  *result = tagres.tags.tags_val;

  return NIS_SUCCESS;
}
libnsl_hidden_nolink_def (nis_stats, GLIBC_2_1)

void
nis_freetags (nis_tag *tags, const int numtags)
{
  int i;

  for (i = 0; i < numtags; ++i)
    free (tags[i].tag_val);
  free (tags);
}
libnsl_hidden_nolink_def (nis_freetags, GLIBC_2_1)
