#include <stdlib.h>
#include <soft-fp.h>
#include <quad.h>

/* Helpers for the Ots functions which receive long double arguments
   in two integer registers, and return values in $16+$17.  */

#define AXP_UNPACK_RAW_Q(X, val)			\
  do {							\
    union _FP_UNION_Q _flo;				\
    _flo.longs.a = val##l;				\
    _flo.longs.b = val##h;				\
    FP_UNPACK_RAW_QP(X, &_flo);				\
  } while (0)

#define AXP_UNPACK_SEMIRAW_Q(X, val)			\
  do {							\
    union _FP_UNION_Q _flo;				\
    _flo.longs.a = val##l;				\
    _flo.longs.b = val##h;				\
    FP_UNPACK_SEMIRAW_QP(X, &_flo);			\
  } while (0)

#define AXP_UNPACK_Q(X, val)				\
  do {							\
    AXP_UNPACK_RAW_Q(X, val);				\
    _FP_UNPACK_CANONICAL(Q, 2, X);			\
  } while (0)

#define AXP_PACK_RAW_Q(val, X) FP_PACK_RAW_QP(&val##_flo, X)

#define AXP_PACK_SEMIRAW_Q(val, X)			\
  do {							\
    _FP_PACK_SEMIRAW(Q, 2, X);				\
    AXP_PACK_RAW_Q(val, X);				\
  } while (0)

#define AXP_PACK_Q(val, X)				\
  do {							\
    _FP_PACK_CANONICAL(Q, 2, X);			\
    AXP_PACK_RAW_Q(val, X);				\
  } while (0)

#define AXP_DECL_RETURN_Q(X) union _FP_UNION_Q X##_flo

/* ??? We don't have a real way to tell the compiler that we're wanting
   to return values in $16+$17.  Instead use a volatile asm to make sure
   that the values are live, and just hope that nothing kills the values
   in between here and the end of the function.  */
#define AXP_RETURN_Q(X)					\
  do {							\
    register long r16 __asm__("16") = X##_flo.longs.a;	\
    register long r17 __asm__("17") = X##_flo.longs.b;	\
    asm volatile ("" : : "r"(r16), "r"(r17));		\
  } while (0)
