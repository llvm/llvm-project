/* Software floating-point emulation.
   Truncate IEEE extended into IEEE half.
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   In addition to the permissions in the GNU Lesser General Public
   License, the Free Software Foundation gives you unlimited
   permission to link the compiled version of this file into
   combinations with other programs, and to distribute those
   combinations without any restriction coming from the use of this
   file.  (The Lesser General Public License restrictions do apply in
   other respects; for example, they cover modification of the file,
   and distribution when not linked into a combine executable.)

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */

#include "soft-fp.h"
#include "half.h"
#include "extended.h"

HFtype
__truncxfhf2 (XFtype a)
{
  FP_DECL_EX;
  FP_DECL_E (A);
  FP_DECL_H (R);
  HFtype r;

  FP_INIT_ROUNDMODE;
  FP_UNPACK_SEMIRAW_E (A, a);
#if _FP_W_TYPE_SIZE < 64
  FP_TRUNC (H, E, 1, 4, R, A);
#else
  FP_TRUNC (H, E, 1, 2, R, A);
#endif
  FP_PACK_SEMIRAW_H (r, R);
  FP_HANDLE_EXCEPTIONS;

  return r;
}
