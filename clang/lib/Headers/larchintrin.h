/*===------------ larchintrin.h - LoongArch intrinsics ---------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef _LOONGARCH_BASE_INTRIN_H
#define _LOONGARCH_BASE_INTRIN_H

#ifdef __cplusplus
extern "C" {
#endif

#if __loongarch_grlen == 64
extern __inline int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __crc_w_b_w(char _1, int _2) {
  return (int)__builtin_loongarch_crc_w_b_w((char)_1, (int)_2);
}

extern __inline int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __crc_w_h_w(short _1, int _2) {
  return (int)__builtin_loongarch_crc_w_h_w((short)_1, (int)_2);
}

extern __inline int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __crc_w_w_w(int _1, int _2) {
  return (int)__builtin_loongarch_crc_w_w_w((int)_1, (int)_2);
}

extern __inline int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __crc_w_d_w(long int _1, int _2) {
  return (int)__builtin_loongarch_crc_w_d_w((long int)_1, (int)_2);
}

extern __inline int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __crcc_w_b_w(char _1, int _2) {
  return (int)__builtin_loongarch_crcc_w_b_w((char)_1, (int)_2);
}

extern __inline int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __crcc_w_h_w(short _1, int _2) {
  return (int)__builtin_loongarch_crcc_w_h_w((short)_1, (int)_2);
}

extern __inline int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __crcc_w_w_w(int _1, int _2) {
  return (int)__builtin_loongarch_crcc_w_w_w((int)_1, (int)_2);
}

extern __inline int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __crcc_w_d_w(long int _1, int _2) {
  return (int)__builtin_loongarch_crcc_w_d_w((long int)_1, (int)_2);
}
#endif

#define __break(/*ui15*/ _1) __builtin_loongarch_break((_1))

#define __dbar(/*ui15*/ _1) __builtin_loongarch_dbar((_1))

#define __ibar(/*ui15*/ _1) __builtin_loongarch_ibar((_1))

#define __syscall(/*ui15*/ _1) __builtin_loongarch_syscall((_1))

#define __csrrd_w(/*ui14*/ _1) ((unsigned int)__builtin_loongarch_csrrd_w((_1)))

#define __csrwr_w(/*unsigned int*/ _1, /*ui14*/ _2)                            \
  ((unsigned int)__builtin_loongarch_csrwr_w((unsigned int)(_1), (_2)))

#define __csrxchg_w(/*unsigned int*/ _1, /*unsigned int*/ _2, /*ui14*/ _3)     \
  ((unsigned int)__builtin_loongarch_csrxchg_w((unsigned int)(_1),             \
                                               (unsigned int)(_2), (_3)))

#if __loongarch_grlen == 64
#define __csrrd_d(/*ui14*/ _1)                                                 \
  ((unsigned long int)__builtin_loongarch_csrrd_d((_1)))

#define __csrwr_d(/*unsigned long int*/ _1, /*ui14*/ _2)                       \
  ((unsigned long int)__builtin_loongarch_csrwr_d((unsigned long int)(_1),     \
                                                  (_2)))

#define __csrxchg_d(/*unsigned long int*/ _1, /*unsigned long int*/ _2,        \
                    /*ui14*/ _3)                                               \
  ((unsigned long int)__builtin_loongarch_csrxchg_d(                           \
      (unsigned long int)(_1), (unsigned long int)(_2), (_3)))
#endif

extern __inline unsigned char
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __iocsrrd_b(unsigned int _1) {
  return (unsigned char)__builtin_loongarch_iocsrrd_b((unsigned int)_1);
}

extern __inline unsigned char
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __iocsrrd_h(unsigned int _1) {
  return (unsigned short)__builtin_loongarch_iocsrrd_h((unsigned int)_1);
}

extern __inline unsigned int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __iocsrrd_w(unsigned int _1) {
  return (unsigned int)__builtin_loongarch_iocsrrd_w((unsigned int)_1);
}

#if __loongarch_grlen == 64
extern __inline unsigned long int
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __iocsrrd_d(unsigned int _1) {
  return (unsigned long int)__builtin_loongarch_iocsrrd_d((unsigned int)_1);
}
#endif

extern __inline void
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __iocsrwr_b(unsigned char _1, unsigned int _2) {
  __builtin_loongarch_iocsrwr_b((unsigned char)_1, (unsigned int)_2);
}

extern __inline void
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __iocsrwr_h(unsigned short _1, unsigned int _2) {
  __builtin_loongarch_iocsrwr_h((unsigned short)_1, (unsigned int)_2);
}

extern __inline void
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __iocsrwr_w(unsigned int _1, unsigned int _2) {
  __builtin_loongarch_iocsrwr_w((unsigned int)_1, (unsigned int)_2);
}

#if __loongarch_grlen == 64
extern __inline void
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    __iocsrwr_d(unsigned long int _1, unsigned int _2) {
  __builtin_loongarch_iocsrwr_d((unsigned long int)_1, (unsigned int)_2);
}
#endif

#ifdef __cplusplus
}
#endif
#endif /* _LOONGARCH_BASE_INTRIN_H */
