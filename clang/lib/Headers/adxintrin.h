/*===---- adxintrin.h - ADX intrinsics -------------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __IMMINTRIN_H
#error "Never use <adxintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __ADXINTRIN_H
#define __ADXINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))

/* Use C++ inline semantics in C++, GNU inline for C mode. */
#if defined(__cplusplus)
#define __INLINE __inline
#else
#define __INLINE static __inline
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* Intrinsics that are available only if __ADX__ defined */
__INLINE unsigned char
    __attribute__((__always_inline__, __nodebug__, __target__("adx")))
    _addcarryx_u32(unsigned char __cf, unsigned int __x, unsigned int __y,
                   unsigned int *__p) {
  return __builtin_ia32_addcarryx_u32(__cf, __x, __y, __p);
}

#ifdef __x86_64__
__INLINE unsigned char
    __attribute__((__always_inline__, __nodebug__, __target__("adx")))
    _addcarryx_u64(unsigned char __cf, unsigned long long __x,
                   unsigned long long __y, unsigned long long *__p) {
  return __builtin_ia32_addcarryx_u64(__cf, __x, __y, __p);
}
#endif

/* Intrinsics that are also available if __ADX__ undefined */
__INLINE unsigned char __DEFAULT_FN_ATTRS _addcarry_u32(unsigned char __cf,
                                                        unsigned int __x,
                                                        unsigned int __y,
                                                        unsigned int *__p) {
  return __builtin_ia32_addcarryx_u32(__cf, __x, __y, __p);
}

#ifdef __x86_64__
__INLINE unsigned char __DEFAULT_FN_ATTRS
_addcarry_u64(unsigned char __cf, unsigned long long __x,
              unsigned long long __y, unsigned long long *__p) {
  return __builtin_ia32_addcarryx_u64(__cf, __x, __y, __p);
}
#endif

__INLINE unsigned char __DEFAULT_FN_ATTRS _subborrow_u32(unsigned char __cf,
                                                         unsigned int __x,
                                                         unsigned int __y,
                                                         unsigned int *__p) {
  return __builtin_ia32_subborrow_u32(__cf, __x, __y, __p);
}

#ifdef __x86_64__
__INLINE unsigned char __DEFAULT_FN_ATTRS
_subborrow_u64(unsigned char __cf, unsigned long long __x,
               unsigned long long __y, unsigned long long *__p) {
  return __builtin_ia32_subborrow_u64(__cf, __x, __y, __p);
}
#endif

#if defined(__cplusplus)
}
#endif

#undef __DEFAULT_FN_ATTRS

#endif /* __ADXINTRIN_H */
