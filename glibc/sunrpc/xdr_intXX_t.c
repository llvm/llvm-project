/* Copyright (c) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1998.

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

#include <rpc/types.h>

/* We play dirty tricks with aliases.  */
#include <rpc/xdr.h>

#include <stdint.h>
#include <shlib-compat.h>

/* XDR 64bit integers */
bool_t
xdr_int64_t (XDR *xdrs, int64_t *ip)
{
  int32_t t1, t2;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      t1 = (int32_t) ((*ip) >> 32);
      t2 = (int32_t) (*ip);
      return (XDR_PUTINT32(xdrs, &t1) && XDR_PUTINT32(xdrs, &t2));
    case XDR_DECODE:
      if (!XDR_GETINT32(xdrs, &t1) || !XDR_GETINT32(xdrs, &t2))
	return FALSE;
      *ip = ((int64_t) t1) << 32;
      *ip |= (uint32_t) t2;	/* Avoid sign extension.  */
      return TRUE;
    case XDR_FREE:
      return TRUE;
    default:
      return FALSE;
    }
}
libc_hidden_nolink_sunrpc (xdr_int64_t, GLIBC_2_1_1)

bool_t
xdr_quad_t (XDR *xdrs, quad_t *ip)
{
  return xdr_int64_t (xdrs, (int64_t *) ip);
}
libc_hidden_nolink_sunrpc (xdr_quad_t, GLIBC_2_3_4)

/* XDR 64bit unsigned integers */
bool_t
xdr_uint64_t (XDR *xdrs, uint64_t *uip)
{
  uint32_t t1;
  uint32_t t2;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      t1 = (uint32_t) ((*uip) >> 32);
      t2 = (uint32_t) (*uip);
      return (XDR_PUTINT32 (xdrs, (int32_t *) &t1) &&
	      XDR_PUTINT32(xdrs, (int32_t *) &t2));
    case XDR_DECODE:
      if (!XDR_GETINT32(xdrs, (int32_t *) &t1) ||
	  !XDR_GETINT32(xdrs, (int32_t *) &t2))
	return FALSE;
      *uip = ((uint64_t) t1) << 32;
      *uip |= t2;
      return TRUE;
    case XDR_FREE:
      return TRUE;
    default:
      return FALSE;
    }
}
libc_hidden_nolink_sunrpc (xdr_uint64_t, GLIBC_2_1_1)

bool_t
xdr_u_quad_t (XDR *xdrs, u_quad_t *ip)
{
  return xdr_uint64_t (xdrs, (uint64_t *) ip);
}
libc_hidden_nolink_sunrpc (xdr_u_quad_t, GLIBC_2_3_4)

/* XDR 32bit integers */
bool_t
xdr_int32_t (XDR *xdrs, int32_t *lp)
{
  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      return XDR_PUTINT32 (xdrs, lp);
    case XDR_DECODE:
      return XDR_GETINT32 (xdrs, lp);
    case XDR_FREE:
      return TRUE;
    default:
      return FALSE;
    }
}
libc_hidden_nolink_sunrpc (xdr_int32_t, GLIBC_2_1)

/* XDR 32bit unsigned integers */
bool_t
xdr_uint32_t (XDR *xdrs, uint32_t *ulp)
{
  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      return XDR_PUTINT32 (xdrs, (int32_t *) ulp);
    case XDR_DECODE:
      return XDR_GETINT32 (xdrs, (int32_t *) ulp);
    case XDR_FREE:
      return TRUE;
    default:
      return FALSE;
    }
}
#ifdef EXPORT_RPC_SYMBOLS
libc_hidden_def (xdr_uint32_t)
#else
libc_hidden_nolink_sunrpc (xdr_uint32_t, GLIBC_2_1)
#endif

/* XDR 16bit integers */
bool_t
xdr_int16_t (XDR *xdrs, int16_t *ip)
{
  int32_t t;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      t = (int32_t) *ip;
      return XDR_PUTINT32 (xdrs, &t);
    case XDR_DECODE:
      if (!XDR_GETINT32 (xdrs, &t))
	return FALSE;
      *ip = (int16_t) t;
      return TRUE;
    case XDR_FREE:
      return TRUE;
    default:
      return FALSE;
    }
}
libc_hidden_nolink_sunrpc (xdr_int16_t, GLIBC_2_1)

/* XDR 16bit unsigned integers */
bool_t
xdr_uint16_t (XDR *xdrs, uint16_t *uip)
{
  uint32_t ut;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      ut = (uint32_t) *uip;
      return XDR_PUTINT32 (xdrs, (int32_t *) &ut);
    case XDR_DECODE:
      if (!XDR_GETINT32 (xdrs, (int32_t *) &ut))
	return FALSE;
      *uip = (uint16_t) ut;
      return TRUE;
    case XDR_FREE:
      return TRUE;
    default:
      return FALSE;
    }
}
libc_hidden_nolink_sunrpc (xdr_uint16_t, GLIBC_2_1)

/* XDR 8bit integers */
bool_t
xdr_int8_t (XDR *xdrs, int8_t *ip)
{
  int32_t t;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      t = (int32_t) *ip;
      return XDR_PUTINT32 (xdrs, &t);
    case XDR_DECODE:
      if (!XDR_GETINT32 (xdrs, &t))
	return FALSE;
      *ip = (int8_t) t;
      return TRUE;
    case XDR_FREE:
      return TRUE;
    default:
      return FALSE;
    }
}
libc_hidden_nolink_sunrpc (xdr_int8_t, GLIBC_2_1)

/* XDR 8bit unsigned integers */
bool_t
xdr_uint8_t (XDR *xdrs, uint8_t *uip)
{
  uint32_t ut;

  switch (xdrs->x_op)
    {
    case XDR_ENCODE:
      ut = (uint32_t) *uip;
      return XDR_PUTINT32 (xdrs, (int32_t *) &ut);
    case XDR_DECODE:
      if (!XDR_GETINT32 (xdrs, (int32_t *) &ut))
	return FALSE;
      *uip = (uint8_t) ut;
      return TRUE;
    case XDR_FREE:
      return TRUE;
    default:
      return FALSE;
    }
}
libc_hidden_nolink_sunrpc (xdr_uint8_t, GLIBC_2_1)
