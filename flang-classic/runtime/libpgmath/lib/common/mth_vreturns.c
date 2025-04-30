/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * These functions are used in returning two vector arguments from a
 * single function.
 *
 * In particular, the sincos() function returns two results, SIN(x), and
 * COS(x).  The compiler expects the return values to be in (x86-64) x/ymm0
 * and x/ymm1.  There is no way with C to return those two values in those
 * two registers without some trickery.
 *
 * Given a function of the form:
 *
 * VectorType
 * FunctionNameReturning2Vectors(vector_type x)
 * {
 *   VectorType sine;
 *   VectorType cossine;
 *   return (sine,cosine); <------------ Will not work.
 * }
 *
 * But, because the our compiler ABI uses the same vector registers
 * to return the two registers as the x86-64 calling sequence ABI,
 * we can call a dummy function with those two "return" values as arguments
 * and then do nothing in the dummy function.
 *
 * This will work as long as the caller does nothing after calling the
 * dummy function.
 *
 * Now function FunctionNameReturning2Vectors becomes:
 *
 * extern VectorType return2VectorType(VectorType, VectorType);
 *
 * VectorType
 * FunctionNameReturning2Vectors(vector_type x)
 * {
 *   VectorType sine;
 *   VectorType cossine;
 *   return (return2VectorType(sine,cosine));
 * }
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

void
__mth_return2vectors(void)
{
    return;
}

#if !defined(TARGET_OSX_X8664) && !defined(TARGET_WIN_X8664)
#if defined(TARGET_ARM64)
#define ALIAS(altname)						\
  void    __mth_return2##altname(void)				\
  __attribute__ ((alias ("__mth_return2vectors")));
#else
/*
 * OSX does not support weak aliases - so just use the generic for all
 * vector types.
 */

#define ALIAS(altname) \
    void    __mth_return2##altname(void) \
        __attribute__ ((weak, alias ("__mth_return2vectors")));
#endif

ALIAS(vrs4_t)
ALIAS(vrd2_t)
ALIAS(vrs8_t)
ALIAS(vrd4_t)
ALIAS(vrs16_t)
ALIAS(vrd8_t)
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
