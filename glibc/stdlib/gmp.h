/* gmp.h -- Definitions for GNU multiple precision functions.

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

#ifndef __GMP_H__

#include <features.h>

#ifndef __GNU_MP__
#define __GNU_MP__ 2
#define __need_size_t
#include <stddef.h>
#undef __need_size_t

#if defined (__STDC__) || defined (__cplusplus)
#define __gmp_const const
#else
#define __gmp_const
#endif

#if defined (__GNUC__)
#define __gmp_inline __inline__
#else
#define __gmp_inline
#endif

#ifndef _EXTERN_INLINE
#ifdef __GNUC__
#define _EXTERN_INLINE __extern_inline
#else
#define _EXTERN_INLINE static
#endif
#endif

#ifdef _SHORT_LIMB
typedef unsigned int		mp_limb_t;
typedef int			mp_limb_signed_t;
#else
#ifdef _LONG_LONG_LIMB
typedef unsigned long long int	mp_limb_t;
typedef long long int		mp_limb_signed_t;
#else
typedef unsigned long int	mp_limb_t;
typedef long int		mp_limb_signed_t;
#endif
#endif

typedef mp_limb_t *		mp_ptr;
typedef __gmp_const mp_limb_t *	mp_srcptr;
typedef long int		mp_size_t;
typedef long int		mp_exp_t;

#ifndef __MP_SMALL__
typedef struct
{
  int _mp_alloc;		/* Number of *limbs* allocated and pointed
				   to by the D field.  */
  int _mp_size;			/* abs(SIZE) is the number of limbs
				   the last field points to.  If SIZE
				   is negative this is a negative
				   number.  */
  mp_limb_t *_mp_d;		/* Pointer to the limbs.  */
} __mpz_struct;
#else
typedef struct
{
  short int _mp_alloc;		/* Number of *limbs* allocated and pointed
				   to by the D field.  */
  short int _mp_size;		/* abs(SIZE) is the number of limbs
				   the last field points to.  If SIZE
				   is negative this is a negative
				   number.  */
  mp_limb_t *_mp_d;		/* Pointer to the limbs.  */
} __mpz_struct;
#endif
#endif /* __GNU_MP__ */

/* User-visible types.  */
typedef __mpz_struct MP_INT;
typedef __mpz_struct mpz_t[1];

/* Structure for rational numbers.  Zero is represented as 0/any, i.e.
   the denominator is ignored.  Negative numbers have the sign in
   the numerator.  */
typedef struct
{
  __mpz_struct _mp_num;
  __mpz_struct _mp_den;
#if 0
  int _mp_num_alloc;		/* Number of limbs allocated
				   for the numerator.  */
  int _mp_num_size;		/* The absolute value of this field is the
				   length of the numerator; the sign is the
				   sign of the entire rational number.  */
  mp_ptr _mp_num;		/* Pointer to the numerator limbs.  */
  int _mp_den_alloc;		/* Number of limbs allocated
				   for the denominator.  */
  int _mp_den_size;		/* Length of the denominator.  (This field
				   should always be positive.) */
  mp_ptr _mp_den;		/* Pointer to the denominator limbs.  */
#endif
} __mpq_struct;

typedef __mpq_struct MP_RAT;
typedef __mpq_struct mpq_t[1];

typedef struct
{
  int _mp_prec;			/* Max precision, in number of `mp_limb_t's.
				   Set by mpf_init and modified by
				   mpf_set_prec.  The area pointed to
				   by the `d' field contains `prec' + 1
				   limbs.  */
  int _mp_size;			/* abs(SIZE) is the number of limbs
				   the last field points to.  If SIZE
				   is negative this is a negative
				   number.  */
  mp_exp_t _mp_exp;		/* Exponent, in the base of `mp_limb_t'.  */
  mp_limb_t *_mp_d;		/* Pointer to the limbs.  */
} __mpf_struct;

/* typedef __mpf_struct MP_FLOAT; */
typedef __mpf_struct mpf_t[1];

/* Types for function declarations in gmp files.  */
/* ??? Should not pollute user name space with these ??? */
typedef __gmp_const __mpz_struct *mpz_srcptr;
typedef __mpz_struct *mpz_ptr;
typedef __gmp_const __mpf_struct *mpf_srcptr;
typedef __mpf_struct *mpf_ptr;
typedef __gmp_const __mpq_struct *mpq_srcptr;
typedef __mpq_struct *mpq_ptr;

#ifndef _PROTO
#if defined (__STDC__) || defined (__cplusplus)
#define _PROTO(x) x
#else
#define _PROTO(x) ()
#endif
#endif

#ifndef __MPN
#if defined (__STDC__) || defined (__cplusplus)
#define __MPN(x) __mpn_##x
#else
#define __MPN(x) __mpn_/**/x
#endif
#endif

#if defined (FILE) || defined (_STDIO_H_) || defined (__STDIO_H__) || defined (H_STDIO)
#define _GMP_H_HAVE_FILE 1
#endif

void mp_set_memory_functions _PROTO ((void *(*) (size_t),
				      void *(*) (void *, size_t, size_t),
				      void (*) (void *, size_t)));
extern const int mp_bits_per_limb;

/**************** Integer (i.e. Z) routines.  ****************/

#if defined (__cplusplus)
extern "C" {
#endif
void *_mpz_realloc _PROTO ((mpz_ptr, mp_size_t));

void mpz_abs _PROTO ((mpz_ptr, mpz_srcptr));
void mpz_add _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_add_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_and _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_array_init _PROTO ((mpz_ptr, mp_size_t, mp_size_t));
void mpz_cdiv_q _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
unsigned long int mpz_cdiv_q_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_cdiv_qr _PROTO ((mpz_ptr, mpz_ptr, mpz_srcptr, mpz_srcptr));
unsigned long int mpz_cdiv_qr_ui _PROTO ((mpz_ptr, mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_cdiv_r _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
unsigned long int mpz_cdiv_r_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
unsigned long int mpz_cdiv_ui _PROTO ((mpz_srcptr, unsigned long int));
void mpz_clear _PROTO ((mpz_ptr));
void mpz_clrbit _PROTO ((mpz_ptr, unsigned long int));
int mpz_cmp _PROTO ((mpz_srcptr, mpz_srcptr));
int mpz_cmp_si _PROTO ((mpz_srcptr, signed long int));
int mpz_cmp_ui _PROTO ((mpz_srcptr, unsigned long int));
void mpz_com _PROTO ((mpz_ptr, mpz_srcptr));
void mpz_divexact _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_fac_ui _PROTO ((mpz_ptr, unsigned long int));
void mpz_fdiv_q _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_fdiv_q_2exp _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
unsigned long int mpz_fdiv_q_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_fdiv_qr _PROTO ((mpz_ptr, mpz_ptr, mpz_srcptr, mpz_srcptr));
unsigned long int mpz_fdiv_qr_ui _PROTO ((mpz_ptr, mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_fdiv_r _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_fdiv_r_2exp _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
unsigned long int mpz_fdiv_r_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
unsigned long int mpz_fdiv_ui _PROTO ((mpz_srcptr, unsigned long int));
void mpz_gcd _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
unsigned long int mpz_gcd_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_gcdext _PROTO ((mpz_ptr, mpz_ptr, mpz_ptr, mpz_srcptr, mpz_srcptr));
/* signed */ long int mpz_get_si _PROTO ((mpz_srcptr));
char *mpz_get_str _PROTO ((char *, int, mpz_srcptr));
unsigned long int mpz_get_ui _PROTO ((mpz_srcptr));
mp_limb_t mpz_getlimbn _PROTO ((mpz_srcptr, mp_size_t));
unsigned long int mpz_hamdist _PROTO ((mpz_srcptr, mpz_srcptr));
void mpz_init _PROTO ((mpz_ptr));
#ifdef _GMP_H_HAVE_FILE
size_t mpz_inp_binary _PROTO ((mpz_ptr, FILE *));
size_t mpz_inp_raw _PROTO ((mpz_ptr, FILE *));
size_t mpz_inp_str _PROTO ((mpz_ptr, FILE *, int));
#endif
void mpz_init_set _PROTO ((mpz_ptr, mpz_srcptr));
void mpz_init_set_d _PROTO ((mpz_ptr, double));
void mpz_init_set_si _PROTO ((mpz_ptr, signed long int));
int mpz_init_set_str _PROTO ((mpz_ptr, const char *, int));
void mpz_init_set_ui _PROTO ((mpz_ptr, unsigned long int));
int mpz_invert _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_ior _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
int mpz_jacobi _PROTO ((mpz_srcptr, mpz_srcptr));
int mpz_legendre _PROTO ((mpz_srcptr, mpz_srcptr));
void mpz_mod _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_mul _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_mul_2exp _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_mul_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_neg _PROTO ((mpz_ptr, mpz_srcptr));
#ifdef _GMP_H_HAVE_FILE
size_t mpz_out_binary _PROTO ((FILE *, mpz_srcptr));
size_t mpz_out_raw _PROTO ((FILE *, mpz_srcptr));
size_t mpz_out_str _PROTO ((FILE *, int, mpz_srcptr));
#endif
int mpz_perfect_square_p _PROTO ((mpz_srcptr));
unsigned long int mpz_popcount _PROTO ((mpz_srcptr));
void mpz_pow_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_powm _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr, mpz_srcptr));
void mpz_powm_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int, mpz_srcptr));
int mpz_probab_prime_p _PROTO ((mpz_srcptr, int));
void mpz_random _PROTO ((mpz_ptr, mp_size_t));
void mpz_random2 _PROTO ((mpz_ptr, mp_size_t));
unsigned long int mpz_scan0 _PROTO ((mpz_srcptr, unsigned long int));
unsigned long int mpz_scan1 _PROTO ((mpz_srcptr, unsigned long int));
void mpz_set _PROTO ((mpz_ptr, mpz_srcptr));
void mpz_set_d _PROTO ((mpz_ptr, double));
void mpz_set_si _PROTO ((mpz_ptr, signed long int));
int mpz_set_str _PROTO ((mpz_ptr, const char *, int));
void mpz_set_ui _PROTO ((mpz_ptr, unsigned long int));
void mpz_setbit _PROTO ((mpz_ptr, unsigned long int));
size_t mpz_size _PROTO ((mpz_srcptr));
size_t mpz_sizeinbase _PROTO ((mpz_srcptr, int));
void mpz_sqrt _PROTO ((mpz_ptr, mpz_srcptr));
void mpz_sqrtrem _PROTO ((mpz_ptr, mpz_ptr, mpz_srcptr));
void mpz_sub _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_sub_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_tdiv_q _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_tdiv_q_2exp _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_tdiv_q_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_tdiv_qr _PROTO ((mpz_ptr, mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_tdiv_qr_ui _PROTO ((mpz_ptr, mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_tdiv_r _PROTO ((mpz_ptr, mpz_srcptr, mpz_srcptr));
void mpz_tdiv_r_2exp _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_tdiv_r_ui _PROTO ((mpz_ptr, mpz_srcptr, unsigned long int));
void mpz_ui_pow_ui _PROTO ((mpz_ptr, unsigned long int, unsigned long int));

/**************** Rational (i.e. Q) routines.  ****************/

void mpq_init _PROTO ((mpq_ptr));
void mpq_clear _PROTO ((mpq_ptr));
void mpq_set _PROTO ((mpq_ptr, mpq_srcptr));
void mpq_set_ui _PROTO ((mpq_ptr, unsigned long int, unsigned long int));
void mpq_set_si _PROTO ((mpq_ptr, signed long int, unsigned long int));
void mpq_add _PROTO ((mpq_ptr, mpq_srcptr, mpq_srcptr));
void mpq_sub _PROTO ((mpq_ptr, mpq_srcptr, mpq_srcptr));
void mpq_mul _PROTO ((mpq_ptr, mpq_srcptr, mpq_srcptr));
void mpq_div _PROTO ((mpq_ptr, mpq_srcptr, mpq_srcptr));
void mpq_neg _PROTO ((mpq_ptr, mpq_srcptr));
int mpq_cmp _PROTO ((mpq_srcptr, mpq_srcptr));
int mpq_cmp_ui _PROTO ((mpq_srcptr, unsigned long int, unsigned long int));
void mpq_inv _PROTO ((mpq_ptr, mpq_srcptr));
void mpq_set_num _PROTO ((mpq_ptr, mpz_srcptr));
void mpq_set_den _PROTO ((mpq_ptr, mpz_srcptr));
void mpq_get_num _PROTO ((mpz_ptr, mpq_srcptr));
void mpq_get_den _PROTO ((mpz_ptr, mpq_srcptr));
double mpq_get_d _PROTO ((mpq_srcptr));
void mpq_canonicalize _PROTO ((mpq_ptr));

/**************** Float (i.e. F) routines.  ****************/

void mpf_abs _PROTO ((mpf_ptr, mpf_srcptr));
void mpf_add _PROTO ((mpf_ptr, mpf_srcptr, mpf_srcptr));
void mpf_add_ui _PROTO ((mpf_ptr, mpf_srcptr, unsigned long int));
void mpf_clear _PROTO ((mpf_ptr));
int mpf_cmp _PROTO ((mpf_srcptr, mpf_srcptr));
int mpf_cmp_si _PROTO ((mpf_srcptr, signed long int));
int mpf_cmp_ui _PROTO ((mpf_srcptr, unsigned long int));
void mpf_div _PROTO ((mpf_ptr, mpf_srcptr, mpf_srcptr));
void mpf_div_2exp _PROTO ((mpf_ptr, mpf_srcptr, unsigned long int));
void mpf_div_ui _PROTO ((mpf_ptr, mpf_srcptr, unsigned long int));
void mpf_dump _PROTO ((mpf_srcptr));
int mpf_eq _PROTO ((mpf_srcptr, mpf_srcptr, unsigned long int));
unsigned long int mpf_get_prec _PROTO ((mpf_srcptr));
char *mpf_get_str _PROTO ((char *, mp_exp_t *, int, size_t, mpf_srcptr));
void mpf_init _PROTO ((mpf_ptr));
void mpf_init2 _PROTO ((mpf_ptr, unsigned long int));
#ifdef _GMP_H_HAVE_FILE
size_t mpf_inp_str _PROTO ((mpf_ptr, FILE *, int));
#endif
void mpf_init_set _PROTO ((mpf_ptr, mpf_srcptr));
void mpf_init_set_d _PROTO ((mpf_ptr, double));
void mpf_init_set_si _PROTO ((mpf_ptr, signed long int));
int mpf_init_set_str _PROTO ((mpf_ptr, char *, int));
void mpf_init_set_ui _PROTO ((mpf_ptr, unsigned long int));
void mpf_mul _PROTO ((mpf_ptr, mpf_srcptr, mpf_srcptr));
void mpf_mul_2exp _PROTO ((mpf_ptr, mpf_srcptr, unsigned long int));
void mpf_mul_ui _PROTO ((mpf_ptr, mpf_srcptr, unsigned long int));
void mpf_neg _PROTO ((mpf_ptr, mpf_srcptr));
#ifdef _GMP_H_HAVE_FILE
size_t mpf_out_str _PROTO ((FILE *, int, size_t, mpf_srcptr));
#endif
void mpf_random2 _PROTO ((mpf_ptr, mp_size_t, mp_exp_t));
void mpf_reldiff _PROTO ((mpf_ptr, mpf_srcptr, mpf_srcptr));
void mpf_set _PROTO ((mpf_ptr, mpf_srcptr));
void mpf_set_d _PROTO ((mpf_ptr, double));
void mpf_set_default_prec _PROTO ((unsigned long int));
void mpf_set_prec _PROTO ((mpf_ptr, unsigned long int));
void mpf_set_prec_raw _PROTO ((mpf_ptr, unsigned long int));
void mpf_set_si _PROTO ((mpf_ptr, signed long int));
int mpf_set_str _PROTO ((mpf_ptr, const char *, int));
void mpf_set_ui _PROTO ((mpf_ptr, unsigned long int));
size_t mpf_size _PROTO ((mpf_srcptr));
void mpf_sqrt _PROTO ((mpf_ptr, mpf_srcptr));
void mpf_sqrt_ui _PROTO ((mpf_ptr, unsigned long int));
void mpf_sub _PROTO ((mpf_ptr, mpf_srcptr, mpf_srcptr));
void mpf_sub_ui _PROTO ((mpf_ptr, mpf_srcptr, unsigned long int));
void mpf_ui_div _PROTO ((mpf_ptr, unsigned long int, mpf_srcptr));
void mpf_ui_sub _PROTO ((mpf_ptr, unsigned long int, mpf_srcptr));
#if defined (__cplusplus)
}
#endif
/************ Low level positive-integer (i.e. N) routines.  ************/

/* This is ugly, but we need to make usr calls reach the prefixed function.  */
#define mpn_add			__MPN(add)
#define mpn_add_1		__MPN(add_1)
#define mpn_add_n		__MPN(add_n)
#define mpn_addmul_1		__MPN(addmul_1)
#define mpn_bdivmod		__MPN(bdivmod)
#define mpn_cmp			__MPN(cmp)
#define mpn_divmod_1		__MPN(divmod_1)
#define mpn_divrem		__MPN(divrem)
#define mpn_divrem_1		__MPN(divrem_1)
#define mpn_dump		__MPN(dump)
#define mpn_gcd			__MPN(gcd)
#define mpn_gcd_1		__MPN(gcd_1)
#define mpn_gcdext		__MPN(gcdext)
#define mpn_get_str		__MPN(get_str)
#define mpn_hamdist		__MPN(hamdist)
#define mpn_lshift		__MPN(lshift)
#define mpn_mod_1		__MPN(mod_1)
#define mpn_mul			__MPN(mul)
#define mpn_mul_1		__MPN(mul_1)
#define mpn_mul_n		__MPN(mul_n)
#define mpn_perfect_square_p	__MPN(perfect_square_p)
#define mpn_popcount		__MPN(popcount)
#define mpn_preinv_mod_1	__MPN(preinv_mod_1)
#define mpn_random2		__MPN(random2)
#define mpn_rshift		__MPN(rshift)
#define mpn_scan0		__MPN(scan0)
#define mpn_scan1		__MPN(scan1)
#define mpn_set_str		__MPN(set_str)
#define mpn_sqrtrem		__MPN(sqrtrem)
#define mpn_sub			__MPN(sub)
#define mpn_sub_1		__MPN(sub_1)
#define mpn_sub_n		__MPN(sub_n)
#define mpn_submul_1		__MPN(submul_1)
#define mpn_udiv_w_sdiv		__MPN(udiv_w_sdiv)

#if defined (__cplusplus)
extern "C" {
#endif
mp_limb_t mpn_add _PROTO ((mp_ptr, mp_srcptr, mp_size_t, mp_srcptr,mp_size_t));
mp_limb_t mpn_add_1 _PROTO ((mp_ptr, mp_srcptr, mp_size_t, mp_limb_t));
mp_limb_t mpn_add_n _PROTO ((mp_ptr, mp_srcptr, mp_srcptr, mp_size_t));
mp_limb_t mpn_addmul_1 _PROTO ((mp_ptr, mp_srcptr, mp_size_t, mp_limb_t));
mp_limb_t mpn_bdivmod _PROTO ((mp_ptr, mp_ptr, mp_size_t, mp_srcptr, mp_size_t, unsigned long int));
int mpn_cmp _PROTO ((mp_srcptr, mp_srcptr, mp_size_t));
mp_limb_t mpn_divmod_1 _PROTO ((mp_ptr, mp_srcptr, mp_size_t, mp_limb_t));
mp_limb_t mpn_divrem _PROTO ((mp_ptr, mp_size_t, mp_ptr, mp_size_t, mp_srcptr, mp_size_t));
mp_limb_t mpn_divrem_1 _PROTO ((mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t));
void mpn_dump _PROTO ((mp_srcptr, mp_size_t));
mp_size_t mpn_gcd _PROTO ((mp_ptr, mp_ptr, mp_size_t, mp_ptr, mp_size_t));
mp_limb_t mpn_gcd_1 _PROTO ((mp_srcptr, mp_size_t, mp_limb_t));
mp_size_t mpn_gcdext _PROTO ((mp_ptr, mp_ptr, mp_ptr, mp_size_t, mp_ptr, mp_size_t));
size_t mpn_get_str _PROTO ((unsigned char *, int, mp_ptr, mp_size_t));
unsigned long int mpn_hamdist _PROTO ((mp_srcptr, mp_srcptr, mp_size_t));
mp_limb_t mpn_lshift _PROTO ((mp_ptr, mp_srcptr, mp_size_t, unsigned int));
mp_limb_t mpn_mod_1 _PROTO ((mp_srcptr, mp_size_t, mp_limb_t));
mp_limb_t mpn_mul _PROTO ((mp_ptr, mp_srcptr, mp_size_t, mp_srcptr, mp_size_t));
mp_limb_t mpn_mul_1 _PROTO ((mp_ptr, mp_srcptr, mp_size_t, mp_limb_t));
void mpn_mul_n _PROTO ((mp_ptr, mp_srcptr, mp_srcptr, mp_size_t));
int mpn_perfect_square_p _PROTO ((mp_srcptr, mp_size_t));
unsigned long int mpn_popcount _PROTO ((mp_srcptr, mp_size_t));
mp_limb_t mpn_preinv_mod_1 _PROTO ((mp_srcptr, mp_size_t, mp_limb_t, mp_limb_t));
void mpn_random2 _PROTO ((mp_ptr, mp_size_t));
mp_limb_t mpn_rshift _PROTO ((mp_ptr, mp_srcptr, mp_size_t, unsigned int));
unsigned long int mpn_scan0 _PROTO ((mp_srcptr, unsigned long int));
unsigned long int mpn_scan1 _PROTO ((mp_srcptr, unsigned long int));
mp_size_t mpn_set_str _PROTO ((mp_ptr, const unsigned char *, size_t, int));
mp_size_t mpn_sqrtrem _PROTO ((mp_ptr, mp_ptr, mp_srcptr, mp_size_t));
mp_limb_t mpn_sub _PROTO ((mp_ptr, mp_srcptr, mp_size_t, mp_srcptr,mp_size_t));
mp_limb_t mpn_sub_1 _PROTO ((mp_ptr, mp_srcptr, mp_size_t, mp_limb_t));
mp_limb_t mpn_sub_n _PROTO ((mp_ptr, mp_srcptr, mp_srcptr, mp_size_t));
mp_limb_t mpn_submul_1 _PROTO ((mp_ptr, mp_srcptr, mp_size_t, mp_limb_t));
#if defined (__cplusplus)
}
#endif

#if defined (__GNUC__) || defined (_FORCE_INLINES)
_EXTERN_INLINE mp_limb_t
#if defined (__STDC__) || defined (__cplusplus)
mpn_add_1 (register mp_ptr res_ptr,
	   register mp_srcptr s1_ptr,
	   register mp_size_t s1_size,
	   register mp_limb_t s2_limb)
#else
mpn_add_1 (res_ptr, s1_ptr, s1_size, s2_limb)
     register mp_ptr res_ptr;
     register mp_srcptr s1_ptr;
     register mp_size_t s1_size;
     register mp_limb_t s2_limb;
#endif
{
  register mp_limb_t x;

  x = *s1_ptr++;
  s2_limb = x + s2_limb;
  *res_ptr++ = s2_limb;
  if (s2_limb < x)
    {
      while (--s1_size != 0)
	{
	  x = *s1_ptr++ + 1;
	  *res_ptr++ = x;
	  if (x != 0)
	    goto fin;
	}

      return 1;
    }

 fin:
  if (res_ptr != s1_ptr)
    {
      mp_size_t i;
      for (i = 0; i < s1_size - 1; i++)
	res_ptr[i] = s1_ptr[i];
    }
  return 0;
}

_EXTERN_INLINE mp_limb_t
#if defined (__STDC__) || defined (__cplusplus)
mpn_add (register mp_ptr res_ptr,
	 register mp_srcptr s1_ptr,
	 register mp_size_t s1_size,
	 register mp_srcptr s2_ptr,
	 register mp_size_t s2_size)
#else
mpn_add (res_ptr, s1_ptr, s1_size, s2_ptr, s2_size)
     register mp_ptr res_ptr;
     register mp_srcptr s1_ptr;
     register mp_size_t s1_size;
     register mp_srcptr s2_ptr;
     register mp_size_t s2_size;
#endif
{
  mp_limb_t cy_limb = 0;

  if (s2_size != 0)
    cy_limb = mpn_add_n (res_ptr, s1_ptr, s2_ptr, s2_size);

  if (s1_size - s2_size != 0)
    cy_limb = mpn_add_1 (res_ptr + s2_size,
			 s1_ptr + s2_size,
			 s1_size - s2_size,
			 cy_limb);
  return cy_limb;
}

_EXTERN_INLINE mp_limb_t
#if defined (__STDC__) || defined (__cplusplus)
mpn_sub_1 (register mp_ptr res_ptr,
	   register mp_srcptr s1_ptr,
	   register mp_size_t s1_size,
	   register mp_limb_t s2_limb)
#else
mpn_sub_1 (res_ptr, s1_ptr, s1_size, s2_limb)
     register mp_ptr res_ptr;
     register mp_srcptr s1_ptr;
     register mp_size_t s1_size;
     register mp_limb_t s2_limb;
#endif
{
  register mp_limb_t x;

  x = *s1_ptr++;
  s2_limb = x - s2_limb;
  *res_ptr++ = s2_limb;
  if (s2_limb > x)
    {
      while (--s1_size != 0)
	{
	  x = *s1_ptr++;
	  *res_ptr++ = x - 1;
	  if (x != 0)
	    goto fin;
	}

      return 1;
    }

 fin:
  if (res_ptr != s1_ptr)
    {
      mp_size_t i;
      for (i = 0; i < s1_size - 1; i++)
	res_ptr[i] = s1_ptr[i];
    }
  return 0;
}

_EXTERN_INLINE mp_limb_t
#if defined (__STDC__) || defined (__cplusplus)
mpn_sub (register mp_ptr res_ptr,
	 register mp_srcptr s1_ptr,
	 register mp_size_t s1_size,
	 register mp_srcptr s2_ptr,
	 register mp_size_t s2_size)
#else
mpn_sub (res_ptr, s1_ptr, s1_size, s2_ptr, s2_size)
     register mp_ptr res_ptr;
     register mp_srcptr s1_ptr;
     register mp_size_t s1_size;
     register mp_srcptr s2_ptr;
     register mp_size_t s2_size;
#endif
{
  mp_limb_t cy_limb = 0;

  if (s2_size != 0)
    cy_limb = mpn_sub_n (res_ptr, s1_ptr, s2_ptr, s2_size);

  if (s1_size - s2_size != 0)
    cy_limb = mpn_sub_1 (res_ptr + s2_size,
			 s1_ptr + s2_size,
			 s1_size - s2_size,
			 cy_limb);
  return cy_limb;
}
#endif /* __GNUC__ */

/* Allow faster testing for negative, zero, and positive.  */
#define mpz_sgn(Z) ((Z)->_mp_size < 0 ? -1 : (Z)->_mp_size > 0)
#define mpf_sgn(F) ((F)->_mp_size < 0 ? -1 : (F)->_mp_size > 0)
#define mpq_sgn(Q) ((Q)->_mp_num._mp_size < 0 ? -1 : (Q)->_mp_num._mp_size > 0)

/* Allow direct user access to numerator and denominator of a mpq_t object.  */
#define mpq_numref(Q) (&((Q)->_mp_num))
#define mpq_denref(Q) (&((Q)->_mp_den))

/* When using GCC, optimize certain common comparisons.  */
#if defined (__GNUC__)
#define mpz_cmp_ui(Z,UI) \
  (__builtin_constant_p (UI) && (UI) == 0				\
   ? mpz_sgn (Z) : mpz_cmp_ui (Z,UI))
#define mpz_cmp_si(Z,UI) \
  (__builtin_constant_p (UI) && (UI) == 0 ? mpz_sgn (Z)			\
   : __builtin_constant_p (UI) && (UI) > 0 ? mpz_cmp_ui (Z,UI)		\
   : mpz_cmp_si (Z,UI))
#define mpq_cmp_ui(Q,NUI,DUI) \
  (__builtin_constant_p (NUI) && (NUI) == 0				\
   ? mpq_sgn (Q) : mpq_cmp_ui (Q,NUI,DUI))
#endif

#define mpn_divmod(qp,np,nsize,dp,dsize) mpn_divrem (qp,0,np,nsize,dp,dsize)
#if 0
#define mpn_divmod_1(qp,np,nsize,dlimb) mpn_divrem_1 (qp,0,np,nsize,dlimb)
#endif

/* Compatibility with GMP 1.  */
#define mpz_mdiv	mpz_fdiv_q
#define mpz_mdivmod	mpz_fdiv_qr
#define mpz_mmod	mpz_fdiv_r
#define mpz_mdiv_ui	mpz_fdiv_q_ui
#define mpz_mdivmod_ui(q,r,n,d) \
  ((r == 0) ? mpz_fdiv_q_ui (q,n,d) : mpz_fdiv_qr_ui (q,r,n,d))
#define mpz_mmod_ui(r,n,d) \
  ((r == 0) ? mpz_fdiv_ui (n,d) : mpz_fdiv_r_ui (r,n,d))

/* Useful synonyms, but not quite compatible with GMP 1.  */
#define mpz_div		mpz_fdiv_q
#define mpz_divmod	mpz_fdiv_qr
#define mpz_div_ui	mpz_fdiv_q_ui
#define mpz_divmod_ui	mpz_fdiv_qr_ui
#define mpz_mod_ui	mpz_fdiv_r_ui
#define mpz_div_2exp	mpz_fdiv_q_2exp
#define mpz_mod_2exp	mpz_fdiv_r_2exp

#define __GNU_MP_VERSION 2
#define __GNU_MP_VERSION_MINOR 0
#define __GMP_H__
#endif /* __GMP_H__ */
