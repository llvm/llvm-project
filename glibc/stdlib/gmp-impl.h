/* Include file for internal GNU MP types and definitions.

Copyright (C) 1991-2021 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MP Library; see the file COPYING.LIB.  If not, see
<https://www.gnu.org/licenses/>.  */

/* When using gcc, make sure to use its builtin alloca.  */
#if ! defined (alloca) && defined (__GNUC__)
#define alloca __builtin_alloca
#define HAVE_ALLOCA
#endif

/* When using cc, do whatever necessary to allow use of alloca.  For many
   machines, this means including alloca.h.  IBM's compilers need a #pragma
   in "each module that needs to use alloca".  */
#if ! defined (alloca)
/* We need lots of variants for MIPS, to cover all versions and perversions
   of OSes for MIPS.  */
#if defined (__mips) || defined (MIPSEL) || defined (MIPSEB) \
 || defined (_MIPSEL) || defined (_MIPSEB) || defined (__sgi) \
 || defined (__alpha) || defined (__sparc) || defined (sparc) \
 || defined (__ksr__)
#include <alloca.h>
#define HAVE_ALLOCA
#endif
#if defined (_IBMR2)
#pragma alloca
#define HAVE_ALLOCA
#endif
#if defined (__DECC)
#define alloca(x) __ALLOCA(x)
#define HAVE_ALLOCA
#endif
#endif

#if (! defined (alloca) && ! defined (HAVE_ALLOCA)) \
    || defined (USE_STACK_ALLOC)
#include "stack-alloc.h"
#else
#define TMP_DECL(m)
#define TMP_ALLOC(x) alloca(x)
#define TMP_MARK(m)
#define TMP_FREE(m)
#endif

#ifndef NULL
#define NULL ((void *) 0)
#endif

#if ! defined (__GNUC__)
#define inline			/* Empty */
#endif

/* Get MAX/MIN macros.  */
#include <sys/param.h>

/* Field access macros.  */
#define SIZ(x) ((x)->_mp_size)
#define PTR(x) ((x)->_mp_d)
#define EXP(x) ((x)->_mp_exp)
#define PREC(x) ((x)->_mp_prec)
#define ALLOC(x) ((x)->_mp_alloc)

#include "gmp-mparam.h"
/* #include "longlong.h" */

#if defined (__STDC__)  || defined (__cplusplus)
void *malloc (size_t);
void *realloc (void *, size_t);
void free (void *);

extern void *	(*_mp_allocate_func) (size_t);
extern void *	(*_mp_reallocate_func) (void *, size_t, size_t);
extern void	(*_mp_free_func) (void *, size_t);

void *_mp_default_allocate (size_t);
void *_mp_default_reallocate (void *, size_t, size_t);
void _mp_default_free (void *, size_t);

#else

#define const			/* Empty */
#define signed			/* Empty */

void *malloc ();
void *realloc ();
void free ();

extern void *	(*_mp_allocate_func) ();
extern void *	(*_mp_reallocate_func) ();
extern void	(*_mp_free_func) ();

void *_mp_default_allocate ();
void *_mp_default_reallocate ();
void _mp_default_free ();
#endif

/* Copy NLIMBS *limbs* from SRC to DST.  */
#define MPN_COPY_INCR(DST, SRC, NLIMBS) \
  do {									\
    mp_size_t __i;							\
    for (__i = 0; __i < (NLIMBS); __i++)				\
      (DST)[__i] = (SRC)[__i];						\
  } while (0)
#define MPN_COPY_DECR(DST, SRC, NLIMBS) \
  do {									\
    mp_size_t __i;							\
    for (__i = (NLIMBS) - 1; __i >= 0; __i--)				\
      (DST)[__i] = (SRC)[__i];						\
  } while (0)
#define MPN_COPY MPN_COPY_INCR

/* Zero NLIMBS *limbs* AT DST.  */
#define MPN_ZERO(DST, NLIMBS) \
  do {									\
    mp_size_t __i;							\
    for (__i = 0; __i < (NLIMBS); __i++)				\
      (DST)[__i] = 0;							\
  } while (0)

#define MPN_NORMALIZE(DST, NLIMBS) \
  do {									\
    while (NLIMBS > 0)							\
      {									\
	if ((DST)[(NLIMBS) - 1] != 0)					\
	  break;							\
	NLIMBS--;							\
      }									\
  } while (0)
#define MPN_NORMALIZE_NOT_ZERO(DST, NLIMBS) \
  do {									\
    while (1)								\
      {									\
	if ((DST)[(NLIMBS) - 1] != 0)					\
	  break;							\
	NLIMBS--;							\
      }									\
  } while (0)

/* Initialize the MP_INT X with space for NLIMBS limbs.
   X should be a temporary variable, and it will be automatically
   cleared out when the running function returns.
   We use __x here to make it possible to accept both mpz_ptr and mpz_t
   arguments.  */
#define MPZ_TMP_INIT(X, NLIMBS) \
  do {									\
    mpz_ptr __x = (X);							\
    __x->_mp_alloc = (NLIMBS);						\
    __x->_mp_d = (mp_ptr) TMP_ALLOC ((NLIMBS) * BYTES_PER_MP_LIMB);	\
  } while (0)

#define MPN_MUL_N_RECURSE(prodp, up, vp, size, tspace) \
  do {									\
    if ((size) < KARATSUBA_THRESHOLD)					\
      impn_mul_n_basecase (prodp, up, vp, size);			\
    else								\
      impn_mul_n (prodp, up, vp, size, tspace);			\
  } while (0);
#define MPN_SQR_N_RECURSE(prodp, up, size, tspace) \
  do {									\
    if ((size) < KARATSUBA_THRESHOLD)					\
      impn_sqr_n_basecase (prodp, up, size);				\
    else								\
      impn_sqr_n (prodp, up, size, tspace);				\
  } while (0);

/* Structure for conversion between internal binary format and
   strings in base 2..36.  */
struct bases
{
  /* Number of digits in the conversion base that always fits in an mp_limb_t.
     For example, for base 10 on a machine where a mp_limb_t has 32 bits this
     is 9, since 10**9 is the largest number that fits into a mp_limb_t.  */
  int chars_per_limb;

  /* log(2)/log(conversion_base) */
  float chars_per_bit_exactly;

  /* base**chars_per_limb, i.e. the biggest number that fits a word, built by
     factors of base.  Exception: For 2, 4, 8, etc, big_base is log2(base),
     i.e. the number of bits used to represent each digit in the base.  */
  mp_limb_t big_base;

  /* A BITS_PER_MP_LIMB bit approximation to 1/big_base, represented as a
     fixed-point number.  Instead of dividing by big_base an application can
     choose to multiply by big_base_inverted.  */
  mp_limb_t big_base_inverted;
};

extern const struct bases __mp_bases[];
extern mp_size_t __gmp_default_fp_limb_precision;

/* Divide the two-limb number in (NH,,NL) by D, with DI being the largest
   limb not larger than (2**(2*BITS_PER_MP_LIMB))/D - (2**BITS_PER_MP_LIMB).
   If this would yield overflow, DI should be the largest possible number
   (i.e., only ones).  For correct operation, the most significant bit of D
   has to be set.  Put the quotient in Q and the remainder in R.  */
#define udiv_qrnnd_preinv(q, r, nh, nl, d, di) \
  do {									\
    mp_limb_t _ql __attribute__ ((unused));				\
    mp_limb_t _q, _r;							\
    mp_limb_t _xh, _xl;							\
    umul_ppmm (_q, _ql, (nh), (di));					\
    _q += (nh);			/* DI is 2**BITS_PER_MP_LIMB too small */\
    umul_ppmm (_xh, _xl, _q, (d));					\
    sub_ddmmss (_xh, _r, (nh), (nl), _xh, _xl);				\
    if (_xh != 0)							\
      {									\
	sub_ddmmss (_xh, _r, _xh, _r, 0, (d));				\
	_q += 1;							\
	if (_xh != 0)							\
	  {								\
	    sub_ddmmss (_xh, _r, _xh, _r, 0, (d));			\
	    _q += 1;							\
	  }								\
      }									\
    if (_r >= (d))							\
      {									\
	_r -= (d);							\
	_q += 1;							\
      }									\
    (r) = _r;								\
    (q) = _q;								\
  } while (0)
/* Like udiv_qrnnd_preinv, but for any value D.  DNORM is D shifted left
   so that its most significant bit is set.  LGUP is ceil(log2(D)).  */
#define udiv_qrnnd_preinv2gen(q, r, nh, nl, d, di, dnorm, lgup) \
  do {									\
    mp_limb_t n2, n10, n1, nadj, q1;					\
    mp_limb_t _xh, _xl;							\
    n2 = ((nh) << (BITS_PER_MP_LIMB - (lgup))) + ((nl) >> 1 >> (l - 1));\
    n10 = (nl) << (BITS_PER_MP_LIMB - (lgup));				\
    n1 = ((mp_limb_signed_t) n10 >> (BITS_PER_MP_LIMB - 1));		\
    nadj = n10 + (n1 & (dnorm));					\
    umul_ppmm (_xh, _xl, di, n2 - n1);					\
    add_ssaaaa (_xh, _xl, _xh, _xl, 0, nadj);				\
    q1 = ~(n2 + _xh);							\
    umul_ppmm (_xh, _xl, q1, d);					\
    add_ssaaaa (_xh, _xl, _xh, _xl, nh, nl);				\
    _xh -= (d);								\
    (r) = _xl + ((d) & _xh);						\
    (q) = _xh - q1;							\
  } while (0)
/* Exactly like udiv_qrnnd_preinv, but branch-free.  It is not clear which
   version to use.  */
#define udiv_qrnnd_preinv2norm(q, r, nh, nl, d, di) \
  do {									\
    mp_limb_t n2, n10, n1, nadj, q1;					\
    mp_limb_t _xh, _xl;							\
    n2 = (nh);								\
    n10 = (nl);								\
    n1 = ((mp_limb_signed_t) n10 >> (BITS_PER_MP_LIMB - 1));		\
    nadj = n10 + (n1 & (d));						\
    umul_ppmm (_xh, _xl, di, n2 - n1);					\
    add_ssaaaa (_xh, _xl, _xh, _xl, 0, nadj);				\
    q1 = ~(n2 + _xh);							\
    umul_ppmm (_xh, _xl, q1, d);					\
    add_ssaaaa (_xh, _xl, _xh, _xl, nh, nl);				\
    _xh -= (d);								\
    (r) = _xl + ((d) & _xh);						\
    (q) = _xh - q1;							\
  } while (0)

#if defined (__GNUC__)
/* Define stuff for longlong.h.  */
typedef unsigned int UQItype	__attribute__ ((mode (QI)));
typedef 	 int SItype	__attribute__ ((mode (SI)));
typedef unsigned int USItype	__attribute__ ((mode (SI)));
typedef		 int DItype	__attribute__ ((mode (DI)));
typedef unsigned int UDItype	__attribute__ ((mode (DI)));
#else
typedef unsigned char UQItype;
typedef 	 long SItype;
typedef unsigned long USItype;
#endif

typedef mp_limb_t UWtype;
typedef unsigned int UHWtype;
#define W_TYPE_SIZE BITS_PER_MP_LIMB

/* Internal mpn calls */
#define impn_mul_n_basecase	__MPN(impn_mul_n_basecase)
#define impn_mul_n		__MPN(impn_mul_n)
#define impn_sqr_n_basecase	__MPN(impn_sqr_n_basecase)
#define impn_sqr_n		__MPN(impn_sqr_n)

#ifndef _PROTO
#if defined (__STDC__) || defined (__cplusplus)
#define _PROTO(x) x
#else
#define _PROTO(x) ()
#endif
#endif

/* Prototypes for internal mpn calls.  */
extern void impn_mul_n_basecase _PROTO ((mp_ptr prodp, mp_srcptr up,
					 mp_srcptr vp, mp_size_t size))
     attribute_hidden;
extern void impn_mul_n _PROTO ((mp_ptr prodp, mp_srcptr up, mp_srcptr vp,
				mp_size_t size, mp_ptr tspace))
     attribute_hidden;
extern void impn_sqr_n_basecase _PROTO ((mp_ptr prodp, mp_srcptr up,
					 mp_size_t size))
     attribute_hidden;
extern void impn_sqr_n _PROTO ((mp_ptr prodp, mp_srcptr up, mp_size_t size,
				mp_ptr tspace))
     attribute_hidden;
