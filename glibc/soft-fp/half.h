/* Software floating-point emulation.
   Definitions for IEEE Half Precision.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
   <https://www.gnu.org/licenses/>.  */

#ifndef SOFT_FP_HALF_H
#define SOFT_FP_HALF_H	1

#if _FP_W_TYPE_SIZE < 32
# error "Here's a nickel kid.  Go buy yourself a real computer."
#endif

#define _FP_FRACTBITS_H		(_FP_W_TYPE_SIZE)

#define _FP_FRACTBITS_DW_H	(_FP_W_TYPE_SIZE)

#define _FP_FRACBITS_H		11
#define _FP_FRACXBITS_H		(_FP_FRACTBITS_H - _FP_FRACBITS_H)
#define _FP_WFRACBITS_H		(_FP_WORKBITS + _FP_FRACBITS_H)
#define _FP_WFRACXBITS_H	(_FP_FRACTBITS_H - _FP_WFRACBITS_H)
#define _FP_EXPBITS_H		5
#define _FP_EXPBIAS_H		15
#define _FP_EXPMAX_H		31

#define _FP_QNANBIT_H		((_FP_W_TYPE) 1 << (_FP_FRACBITS_H-2))
#define _FP_QNANBIT_SH_H	((_FP_W_TYPE) 1 << (_FP_FRACBITS_H-2+_FP_WORKBITS))
#define _FP_IMPLBIT_H		((_FP_W_TYPE) 1 << (_FP_FRACBITS_H-1))
#define _FP_IMPLBIT_SH_H	((_FP_W_TYPE) 1 << (_FP_FRACBITS_H-1+_FP_WORKBITS))
#define _FP_OVERFLOW_H		((_FP_W_TYPE) 1 << (_FP_WFRACBITS_H))

#define _FP_WFRACBITS_DW_H	(2 * _FP_WFRACBITS_H)
#define _FP_WFRACXBITS_DW_H	(_FP_FRACTBITS_DW_H - _FP_WFRACBITS_DW_H)
#define _FP_HIGHBIT_DW_H	\
  ((_FP_W_TYPE) 1 << (_FP_WFRACBITS_DW_H - 1) % _FP_W_TYPE_SIZE)

/* The implementation of _FP_MUL_MEAT_H and _FP_DIV_MEAT_H should be
   chosen by the target machine.  */

typedef float HFtype __attribute__ ((mode (HF)));

union _FP_UNION_H
{
  HFtype flt;
  struct _FP_STRUCT_LAYOUT
  {
#if __BYTE_ORDER == __BIG_ENDIAN
    unsigned sign : 1;
    unsigned exp  : _FP_EXPBITS_H;
    unsigned frac : _FP_FRACBITS_H - (_FP_IMPLBIT_H != 0);
#else
    unsigned frac : _FP_FRACBITS_H - (_FP_IMPLBIT_H != 0);
    unsigned exp  : _FP_EXPBITS_H;
    unsigned sign : 1;
#endif
  } bits;
};

#define FP_DECL_H(X)		_FP_DECL (1, X)
#define FP_UNPACK_RAW_H(X, val)	_FP_UNPACK_RAW_1 (H, X, (val))
#define FP_UNPACK_RAW_HP(X, val)	_FP_UNPACK_RAW_1_P (H, X, (val))
#define FP_PACK_RAW_H(val, X)	_FP_PACK_RAW_1 (H, (val), X)
#define FP_PACK_RAW_HP(val, X)			\
  do						\
    {						\
      if (!FP_INHIBIT_RESULTS)			\
	_FP_PACK_RAW_1_P (H, (val), X);		\
    }						\
  while (0)

#define FP_UNPACK_H(X, val)			\
  do						\
    {						\
      _FP_UNPACK_RAW_1 (H, X, (val));		\
      _FP_UNPACK_CANONICAL (H, 1, X);		\
    }						\
  while (0)

#define FP_UNPACK_HP(X, val)			\
  do						\
    {						\
      _FP_UNPACK_RAW_1_P (H, X, (val));		\
      _FP_UNPACK_CANONICAL (H, 1, X);		\
    }						\
  while (0)

#define FP_UNPACK_SEMIRAW_H(X, val)		\
  do						\
    {						\
      _FP_UNPACK_RAW_1 (H, X, (val));		\
      _FP_UNPACK_SEMIRAW (H, 1, X);		\
    }						\
  while (0)

#define FP_UNPACK_SEMIRAW_HP(X, val)		\
  do						\
    {						\
      _FP_UNPACK_RAW_1_P (H, X, (val));		\
      _FP_UNPACK_SEMIRAW (H, 1, X);		\
    }						\
  while (0)

#define FP_PACK_H(val, X)			\
  do						\
    {						\
      _FP_PACK_CANONICAL (H, 1, X);		\
      _FP_PACK_RAW_1 (H, (val), X);		\
    }						\
  while (0)

#define FP_PACK_HP(val, X)			\
  do						\
    {						\
      _FP_PACK_CANONICAL (H, 1, X);		\
      if (!FP_INHIBIT_RESULTS)			\
	_FP_PACK_RAW_1_P (H, (val), X);		\
    }						\
  while (0)

#define FP_PACK_SEMIRAW_H(val, X)		\
  do						\
    {						\
      _FP_PACK_SEMIRAW (H, 1, X);		\
      _FP_PACK_RAW_1 (H, (val), X);		\
    }						\
  while (0)

#define FP_PACK_SEMIRAW_HP(val, X)		\
  do						\
    {						\
      _FP_PACK_SEMIRAW (H, 1, X);		\
      if (!FP_INHIBIT_RESULTS)			\
	_FP_PACK_RAW_1_P (H, (val), X);		\
    }						\
  while (0)

#define FP_TO_INT_H(r, X, rsz, rsg)	_FP_TO_INT (H, 1, (r), X, (rsz), (rsg))
#define FP_TO_INT_ROUND_H(r, X, rsz, rsg)	\
  _FP_TO_INT_ROUND (H, 1, (r), X, (rsz), (rsg))
#define FP_FROM_INT_H(X, r, rs, rt)	_FP_FROM_INT (H, 1, X, (r), (rs), rt)

/* HFmode arithmetic is not implemented.  */

#define _FP_FRAC_HIGH_H(X)	_FP_FRAC_HIGH_1 (X)
#define _FP_FRAC_HIGH_RAW_H(X)	_FP_FRAC_HIGH_1 (X)
#define _FP_FRAC_HIGH_DW_H(X)	_FP_FRAC_HIGH_1 (X)

#define FP_CMP_EQ_H(r, X, Y, ex)       _FP_CMP_EQ (H, 1, (r), X, Y, (ex))

#endif /* !SOFT_FP_HALF_H */
