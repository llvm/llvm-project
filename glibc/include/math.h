#ifndef	_MATH_H

#ifdef _ISOMAC
# undef NO_LONG_DOUBLE
#endif

#include <math/math.h>

#ifndef _ISOMAC
/* Now define the internal interfaces.  */
extern int __signgam;

# if IS_IN (libc) || IS_IN (libm)
hidden_proto (__finite)
hidden_proto (__isinf)
hidden_proto (__isnan)
hidden_proto (__finitef)
hidden_proto (__isinff)
hidden_proto (__isnanf)

#  if !defined __NO_LONG_DOUBLE_MATH \
      && __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 0
hidden_proto (__finitel)
hidden_proto (__isinfl)
hidden_proto (__isnanl)
#  endif

#  if __HAVE_DISTINCT_FLOAT128
hidden_proto (__finitef128)
hidden_proto (__isinff128)
hidden_proto (__isnanf128)
#  endif
# endif

libm_hidden_proto (__fpclassify)
libm_hidden_proto (__fpclassifyf)
libm_hidden_proto (__issignaling)
libm_hidden_proto (__issignalingf)
libm_hidden_proto (__exp)
libm_hidden_proto (__expf)

#  if !defined __NO_LONG_DOUBLE_MATH \
      && __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 0
libm_hidden_proto (__fpclassifyl)
libm_hidden_proto (__issignalingl)
libm_hidden_proto (__expl)
libm_hidden_proto (__expm1l)
# endif

# if __HAVE_DISTINCT_FLOAT128
libm_hidden_proto (__fpclassifyf128)
libm_hidden_proto (__issignalingf128)
libm_hidden_proto (__expf128)
libm_hidden_proto (__expm1f128)
# endif

#include <stdint.h>
#include <nan-high-order-bit.h>

/* A union which permits us to convert between a float and a 32 bit
   int.  */

typedef union
{
  float value;
  uint32_t word;
} ieee_float_shape_type;

/* Get a 32 bit int from a float.  */
#ifndef GET_FLOAT_WORD
# define GET_FLOAT_WORD(i,d)					\
do {								\
  ieee_float_shape_type gf_u;					\
  gf_u.value = (d);						\
  (i) = gf_u.word;						\
} while (0)
#endif

/* Set a float from a 32 bit int.  */
#ifndef SET_FLOAT_WORD
# define SET_FLOAT_WORD(d,i)					\
do {								\
  ieee_float_shape_type sf_u;					\
  sf_u.word = (i);						\
  (d) = sf_u.value;						\
} while (0)
#endif

extern inline int
__issignalingf (float x)
{
  uint32_t xi;
  GET_FLOAT_WORD (xi, x);
#if HIGH_ORDER_BIT_IS_SET_FOR_SNAN
  /* We only have to care about the high-order bit of x's significand, because
     having it set (sNaN) already makes the significand different from that
     used to designate infinity.  */
  return (xi & 0x7fc00000) == 0x7fc00000;
#else
  /* To keep the following comparison simple, toggle the quiet/signaling bit,
     so that it is set for sNaNs.  This is inverse to IEEE 754-2008 (as well as
     common practice for IEEE 754-1985).  */
  xi ^= 0x00400000;
  /* We have to compare for greater (instead of greater or equal), because x's
     significand being all-zero designates infinity not NaN.  */
  return (xi & 0x7fffffff) > 0x7fc00000;
#endif
}

# if __HAVE_DISTINCT_FLOAT128

/* __builtin_isinf_sign is broken in GCC < 7 for float128.  */
#  if ! __GNUC_PREREQ (7, 0)
#   include <ieee754_float128.h>
extern inline int
__isinff128 (_Float128 x)
{
  int64_t hx, lx;
  GET_FLOAT128_WORDS64 (hx, lx, x);
  lx |= (hx & 0x7fffffffffffffffLL) ^ 0x7fff000000000000LL;
  lx |= -lx;
  return ~(lx >> 63) & (hx >> 62);
}
#  endif

extern inline _Float128
fabsf128 (_Float128 x)
{
  return __builtin_fabsf128 (x);
}
# endif

# if !(defined __FINITE_MATH_ONLY__ && __FINITE_MATH_ONLY__ > 0)
#  ifndef NO_MATH_REDIRECT
/* Declare some functions for use within GLIBC.  Compilers typically
   inline those functions as a single instruction.  Use an asm to
   avoid use of PLTs if it doesn't.  */
#   define MATH_REDIRECT(FUNC, PREFIX, ARGS)			\
  float (FUNC ## f) (ARGS (float)) asm (PREFIX #FUNC "f");	\
  double (FUNC) (ARGS (double)) asm (PREFIX #FUNC );		\
  MATH_REDIRECT_LDBL (FUNC, PREFIX, ARGS)			\
  MATH_REDIRECT_F128 (FUNC, PREFIX, ARGS)
#   if defined __NO_LONG_DOUBLE_MATH 				\
       || __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 1
#    define MATH_REDIRECT_LDBL(FUNC, PREFIX, ARGS)
#   else
#    define MATH_REDIRECT_LDBL(FUNC, PREFIX, ARGS)			\
  long double (FUNC ## l) (ARGS (long double)) asm (PREFIX #FUNC "l");
#   endif
#   if __HAVE_DISTINCT_FLOAT128
#    define MATH_REDIRECT_F128(FUNC, PREFIX, ARGS)			\
  _Float128 (FUNC ## f128) (ARGS (_Float128)) asm (PREFIX #FUNC "f128");
#   else
#    define MATH_REDIRECT_F128(FUNC, PREFIX, ARGS)
#   endif
#   define MATH_REDIRECT_UNARY_ARGS(TYPE) TYPE
#   define MATH_REDIRECT_BINARY_ARGS(TYPE) TYPE, TYPE
MATH_REDIRECT (sqrt, "__ieee754_", MATH_REDIRECT_UNARY_ARGS)
MATH_REDIRECT (ceil, "__", MATH_REDIRECT_UNARY_ARGS)
MATH_REDIRECT (floor, "__", MATH_REDIRECT_UNARY_ARGS)
MATH_REDIRECT (roundeven, "__", MATH_REDIRECT_UNARY_ARGS)
MATH_REDIRECT (rint, "__", MATH_REDIRECT_UNARY_ARGS)
MATH_REDIRECT (trunc, "__", MATH_REDIRECT_UNARY_ARGS)
MATH_REDIRECT (round, "__", MATH_REDIRECT_UNARY_ARGS)
MATH_REDIRECT (copysign, "__", MATH_REDIRECT_BINARY_ARGS)
#  endif
# endif

#endif
#endif
