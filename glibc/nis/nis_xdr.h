/* Copyright (c) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@suse.de>, 1998.

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

#ifndef __NIS_XDR_H
#define __NIS_XDR_H 1

#include <features.h>

extern  bool_t _xdr_nis_attr (XDR *, nis_attr*) attribute_hidden;
extern  bool_t _xdr_nis_name (XDR *, nis_name*) attribute_hidden;
extern  bool_t _xdr_nis_server (XDR *, nis_server*) attribute_hidden;
extern  bool_t _xdr_directory_obj (XDR *, directory_obj*) attribute_hidden;
extern  bool_t _xdr_nis_object (XDR *, nis_object*) attribute_hidden;
extern  bool_t _xdr_nis_error (XDR *, nis_error*) attribute_hidden;
extern  bool_t _xdr_ns_request (XDR *, ns_request*) attribute_hidden;
extern  bool_t _xdr_ping_args (XDR *, ping_args*) attribute_hidden;
extern  bool_t _xdr_cp_result (XDR *, cp_result*) attribute_hidden;
extern  bool_t _xdr_nis_tag (XDR *, nis_tag*) attribute_hidden;
extern  bool_t _xdr_nis_taglist (XDR *, nis_taglist*) attribute_hidden;
extern  bool_t _xdr_fd_args (XDR *, fd_args*) attribute_hidden;
extern  bool_t _xdr_fd_result (XDR *, fd_result*) attribute_hidden;

extern  bool_t _xdr_ib_request (XDR *, ib_request*);
libnsl_hidden_proto (_xdr_ib_request)
extern  bool_t _xdr_nis_result (XDR *, nis_result*);
libnsl_hidden_proto (_xdr_nis_result)

#endif
