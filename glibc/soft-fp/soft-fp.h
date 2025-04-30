/* Software floating-point emulation.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@cygnus.com),
		  Jakub Jelinek (jj@ultra.linux.cz),
		  David S. Miller (davem@redhat.com) and
		  Peter Maydell (pmaydell@chiark.greenend.org.uk).

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

#ifndef SOFT_FP_H
#define SOFT_FP_H	1

#ifdef _LIBC
# include <sfp-machine.h>
#elif defined __KERNEL__
/* The Linux kernel uses asm/ names for architecture-specific
   files.  */
# include <asm/sfp-machine.h>
#else
# include "sfp-machine.h"
#endif

/* Allow sfp-machine to have its own byte order definitions.  */
#ifndef __BYTE_ORDER
# ifdef _LIBC
#  include <endian.h>
# else
#  error "endianness not defined by sfp-machine.h"
# endif
#endif

/* For unreachable default cases in switch statements over bitwise OR
   of FP_CLS_* values.  */
#if (defined __GNUC__							\
     && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5)))
# define _FP_UNREACHABLE	__builtin_unreachable ()
#else
# define _FP_UNREACHABLE	abort ()
#endif

#if ((defined __GNUC__							\
      && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)))	\
     || (defined __STDC_VERSION__ && __STDC_VERSION__ >= 201112L))
# define _FP_STATIC_ASSERT(expr, msg)		\
  _Static_assert ((expr), msg)
#else
# define _FP_STATIC_ASSERT(expr, msg)					\
  extern int (*__Static_assert_function (void))				\
    [!!sizeof (struct { int __error_if_negative: (expr) ? 2 : -1; })]
#endif

/* In the Linux kernel, some architectures have a single function that
   uses different kinds of unpacking and packing depending on the
   instruction being emulated, meaning it is not readily visible to
   the compiler that variables from _FP_DECL and _FP_FRAC_DECL_*
   macros are only used in cases where they were initialized.  */
#ifdef __KERNEL__
# define _FP_ZERO_INIT		= 0
#else
# define _FP_ZERO_INIT
#endif

#define _FP_WORKBITS		3
#define _FP_WORK_LSB		((_FP_W_TYPE) 1 << 3)
#define _FP_WORK_ROUND		((_FP_W_TYPE) 1 << 2)
#define _FP_WORK_GUARD		((_FP_W_TYPE) 1 << 1)
#define _FP_WORK_STICKY		((_FP_W_TYPE) 1 << 0)

#ifndef FP_RND_NEAREST
# define FP_RND_NEAREST		0
# define FP_RND_ZERO		1
# define FP_RND_PINF		2
# define FP_RND_MINF		3
#endif
#ifndef FP_ROUNDMODE
# define FP_ROUNDMODE		FP_RND_NEAREST
#endif

/* By default don't care about exceptions.  */
#ifndef FP_EX_INVALID
# define FP_EX_INVALID		0
#endif
#ifndef FP_EX_OVERFLOW
# define FP_EX_OVERFLOW		0
#endif
#ifndef FP_EX_UNDERFLOW
# define FP_EX_UNDERFLOW	0
#endif
#ifndef FP_EX_DIVZERO
# define FP_EX_DIVZERO		0
#endif
#ifndef FP_EX_INEXACT
# define FP_EX_INEXACT		0
#endif
#ifndef FP_EX_DENORM
# define FP_EX_DENORM		0
#endif

/* Sub-exceptions of "invalid".  */
/* Signaling NaN operand.  */
#ifndef FP_EX_INVALID_SNAN
# define FP_EX_INVALID_SNAN	0
#endif
/* Inf * 0.  */
#ifndef FP_EX_INVALID_IMZ
# define FP_EX_INVALID_IMZ	0
#endif
/* fma (Inf, 0, c).  */
#ifndef FP_EX_INVALID_IMZ_FMA
# define FP_EX_INVALID_IMZ_FMA	0
#endif
/* Inf - Inf.  */
#ifndef FP_EX_INVALID_ISI
# define FP_EX_INVALID_ISI	0
#endif
/* 0 / 0.  */
#ifndef FP_EX_INVALID_ZDZ
# define FP_EX_INVALID_ZDZ	0
#endif
/* Inf / Inf.  */
#ifndef FP_EX_INVALID_IDI
# define FP_EX_INVALID_IDI	0
#endif
/* sqrt (negative).  */
#ifndef FP_EX_INVALID_SQRT
# define FP_EX_INVALID_SQRT	0
#endif
/* Invalid conversion to integer.  */
#ifndef FP_EX_INVALID_CVI
# define FP_EX_INVALID_CVI	0
#endif
/* Invalid comparison.  */
#ifndef FP_EX_INVALID_VC
# define FP_EX_INVALID_VC	0
#endif

/* _FP_STRUCT_LAYOUT may be defined as an attribute to determine the
   struct layout variant used for structures where bit-fields are used
   to access specific parts of binary floating-point numbers.  This is
   required for systems where the default ABI uses struct layout with
   differences in how consecutive bit-fields are laid out from the
   default expected by soft-fp.  */
#ifndef _FP_STRUCT_LAYOUT
# define _FP_STRUCT_LAYOUT
#endif

#ifdef _FP_DECL_EX
# define FP_DECL_EX					\
  int _fex = 0;						\
  _FP_DECL_EX
#else
# define FP_DECL_EX int _fex = 0
#endif

/* Initialize any machine-specific state used in FP_ROUNDMODE,
   FP_TRAPPING_EXCEPTIONS or FP_HANDLE_EXCEPTIONS.  */
#ifndef FP_INIT_ROUNDMODE
# define FP_INIT_ROUNDMODE do {} while (0)
#endif

/* Initialize any machine-specific state used in
   FP_TRAPPING_EXCEPTIONS or FP_HANDLE_EXCEPTIONS.  */
#ifndef FP_INIT_TRAPPING_EXCEPTIONS
# define FP_INIT_TRAPPING_EXCEPTIONS FP_INIT_ROUNDMODE
#endif

/* Initialize any machine-specific state used in
   FP_HANDLE_EXCEPTIONS.  */
#ifndef FP_INIT_EXCEPTIONS
# define FP_INIT_EXCEPTIONS FP_INIT_TRAPPING_EXCEPTIONS
#endif

#ifndef FP_HANDLE_EXCEPTIONS
# define FP_HANDLE_EXCEPTIONS do {} while (0)
#endif

/* Whether to flush subnormal inputs to zero with the same sign.  */
#ifndef FP_DENORM_ZERO
# define FP_DENORM_ZERO 0
#endif

#ifndef FP_INHIBIT_RESULTS
/* By default we write the results always.
   sfp-machine may override this and e.g.
   check if some exceptions are unmasked
   and inhibit it in such a case.  */
# define FP_INHIBIT_RESULTS 0
#endif

#define FP_SET_EXCEPTION(ex)				\
  _fex |= (ex)

#define FP_CUR_EXCEPTIONS				\
  (_fex)

#ifndef FP_TRAPPING_EXCEPTIONS
# define FP_TRAPPING_EXCEPTIONS 0
#endif

/* A file using soft-fp may define FP_NO_EXCEPTIONS before including
   soft-fp.h to indicate that, although a macro used there could raise
   exceptions, or do rounding and potentially thereby raise
   exceptions, for some arguments, for the particular arguments used
   in that file no exceptions or rounding can occur.  Such a file
   should not itself use macros relating to handling exceptions and
   rounding modes; this is only for indirect uses (in particular, in
   _FP_FROM_INT and the macros it calls).  */
#ifdef FP_NO_EXCEPTIONS

# undef FP_SET_EXCEPTION
# define FP_SET_EXCEPTION(ex) do {} while (0)

# undef FP_CUR_EXCEPTIONS
# define FP_CUR_EXCEPTIONS 0

# undef FP_TRAPPING_EXCEPTIONS
# define FP_TRAPPING_EXCEPTIONS 0

# undef FP_ROUNDMODE
# define FP_ROUNDMODE FP_RND_ZERO

# undef _FP_TININESS_AFTER_ROUNDING
# define _FP_TININESS_AFTER_ROUNDING 0

#endif

/* A file using soft-fp may define FP_NO_EXACT_UNDERFLOW before
   including soft-fp.h to indicate that, although a macro used there
   could allow for the case of exact underflow requiring the underflow
   exception to be raised if traps are enabled, for the particular
   arguments used in that file no exact underflow can occur.  */
#ifdef FP_NO_EXACT_UNDERFLOW
# undef FP_TRAPPING_EXCEPTIONS
# define FP_TRAPPING_EXCEPTIONS 0
#endif

#define _FP_ROUND_NEAREST(wc, X)				\
  do								\
    {								\
      if ((_FP_FRAC_LOW_##wc (X) & 15) != _FP_WORK_ROUND)	\
	_FP_FRAC_ADDI_##wc (X, _FP_WORK_ROUND);			\
    }								\
  while (0)

#define _FP_ROUND_ZERO(wc, X)		(void) 0

#define _FP_ROUND_PINF(wc, X)				\
  do							\
    {							\
      if (!X##_s && (_FP_FRAC_LOW_##wc (X) & 7))	\
	_FP_FRAC_ADDI_##wc (X, _FP_WORK_LSB);		\
    }							\
  while (0)

#define _FP_ROUND_MINF(wc, X)			\
  do						\
    {						\
      if (X##_s && (_FP_FRAC_LOW_##wc (X) & 7))	\
	_FP_FRAC_ADDI_##wc (X, _FP_WORK_LSB);	\
    }						\
  while (0)

#define _FP_ROUND(wc, X)			\
  do						\
    {						\
      if (_FP_FRAC_LOW_##wc (X) & 7)		\
	{					\
	  FP_SET_EXCEPTION (FP_EX_INEXACT);	\
	  switch (FP_ROUNDMODE)			\
	    {					\
	    case FP_RND_NEAREST:		\
	      _FP_ROUND_NEAREST (wc, X);	\
	      break;				\
	    case FP_RND_ZERO:			\
	      _FP_ROUND_ZERO (wc, X);		\
	      break;				\
	    case FP_RND_PINF:			\
	      _FP_ROUND_PINF (wc, X);		\
	      break;				\
	    case FP_RND_MINF:			\
	      _FP_ROUND_MINF (wc, X);		\
	      break;				\
	    }					\
	}					\
    }						\
  while (0)

#define FP_CLS_NORMAL		0
#define FP_CLS_ZERO		1
#define FP_CLS_INF		2
#define FP_CLS_NAN		3

#define _FP_CLS_COMBINE(x, y)	(((x) << 2) | (y))

#include "op-1.h"
#include "op-2.h"
#include "op-4.h"
#include "op-8.h"
#include "op-common.h"

/* Sigh.  Silly things longlong.h needs.  */
#define UWtype		_FP_W_TYPE
#define W_TYPE_SIZE	_FP_W_TYPE_SIZE

typedef int QItype __attribute__ ((mode (QI)));
typedef int SItype __attribute__ ((mode (SI)));
typedef int DItype __attribute__ ((mode (DI)));
typedef unsigned int UQItype __attribute__ ((mode (QI)));
typedef unsigned int USItype __attribute__ ((mode (SI)));
typedef unsigned int UDItype __attribute__ ((mode (DI)));
#if _FP_W_TYPE_SIZE == 32
typedef unsigned int UHWtype __attribute__ ((mode (HI)));
#elif _FP_W_TYPE_SIZE == 64
typedef USItype UHWtype;
#endif

#ifndef CMPtype
# define CMPtype	int
#endif

#define SI_BITS		(__CHAR_BIT__ * (int) sizeof (SItype))
#define DI_BITS		(__CHAR_BIT__ * (int) sizeof (DItype))

#ifndef umul_ppmm
# ifdef _LIBC
#  include <stdlib/longlong.h>
# else
#  include "longlong.h"
# endif
#endif

#endif /* !SOFT_FP_H */
