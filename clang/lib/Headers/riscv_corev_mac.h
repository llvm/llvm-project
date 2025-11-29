/*===---- riscv_corev_mac.h - CORE-V multiply accumulate intrinsics --------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_COREV_MAC_H
#define __RISCV_COREV_MAC_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__riscv_xcvmac)

#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_mac_mac(long a, long b,
                                                             long c) {
  return __builtin_riscv_cv_mac_mac(a, b, c);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_mac_msu(long a, long b,
                                                             long c) {
  return __builtin_riscv_cv_mac_msu(a, b, c);
}

#define __riscv_cv_mac_muluN(rs1, rs2, SHIFT)                                  \
  (unsigned long)__builtin_riscv_cv_mac_muluN(                                 \
      (unsigned long)(rs1), (unsigned long)(rs2), (const uint8_t)(SHIFT))

#define __riscv_cv_mac_mulhhuN(rs1, rs2, SHIFT)                                \
  (unsigned long)__builtin_riscv_cv_mac_mulhhuN(                               \
      (unsigned long)(rs1), (unsigned long)(rs2), (const uint8_t)(SHIFT))

#define __riscv_cv_mac_mulsN(rs1, rs2, SHIFT)                                  \
  (long)__builtin_riscv_cv_mac_mulsN(                                          \
      (unsigned long)(rs1), (unsigned long)(rs2), (const uint8_t)(SHIFT))

#define __riscv_cv_mac_mulhhsN(rs1, rs2, SHIFT)                                \
  (long)__builtin_riscv_cv_mac_mulhhsN(                                        \
      (unsigned long)(rs1), (unsigned long)(rs2), (const uint8_t)(SHIFT))

#define __riscv_cv_mac_muluRN(rs1, rs2, SHIFT)                                 \
  (unsigned long)__builtin_riscv_cv_mac_muluRN(                                \
      (unsigned long)(rs1), (unsigned long)(rs2), (const uint8_t)(SHIFT))

#define __riscv_cv_mac_mulhhuRN(rs1, rs2, SHIFT)                               \
  (unsigned long)__builtin_riscv_cv_mac_mulhhuRN(                              \
      (unsigned long)(rs1), (unsigned long)(rs2), (const uint8_t)(SHIFT))

#define __riscv_cv_mac_mulsRN(rs1, rs2, SHIFT)                                 \
  (long)__builtin_riscv_cv_mac_mulsRN(                                         \
      (unsigned long)(rs1), (unsigned long)(rs2), (const uint8_t)(SHIFT))

#define __riscv_cv_mac_mulhhsRN(rs1, rs2, SHIFT)                               \
  (long)__builtin_riscv_cv_mac_mulhhsRN(                                       \
      (unsigned long)(rs1), (unsigned long)(rs2), (const uint8_t)(SHIFT))

#define __riscv_cv_mac_macuN(rs1, rs2, rD, SHIFT)                              \
  (unsigned long)__builtin_riscv_cv_mac_macuN(                                 \
      (unsigned long)(rs1), (unsigned long)(rs2), (unsigned long)(rD),         \
      (const uint8_t)(SHIFT))

#define __riscv_cv_mac_machhuN(rs1, rs2, rD, SHIFT)                            \
  (unsigned long)__builtin_riscv_cv_mac_machhuN(                               \
      (unsigned long)(rs1), (unsigned long)(rs2), (unsigned long)(rD),         \
      (const uint8_t)(SHIFT))

#define __riscv_cv_mac_macsN(rs1, rs2, rD, SHIFT)                              \
  (long)__builtin_riscv_cv_mac_macsN(                                          \
      (unsigned long)(rs1), (unsigned long)(rs2), (unsigned long)(rD),         \
      (const uint8_t)(SHIFT))

#define __riscv_cv_mac_machhsN(rs1, rs2, rD, SHIFT)                            \
  (long)__builtin_riscv_cv_mac_machhsN(                                        \
      (unsigned long)(rs1), (unsigned long)(rs2), (unsigned long)(rD),         \
      (const uint8_t)(SHIFT))

#define __riscv_cv_mac_macuRN(rs1, rs2, rD, SHIFT)                             \
  (unsigned long)__builtin_riscv_cv_mac_macuRN(                                \
      (unsigned long)(rs1), (unsigned long)(rs2), (unsigned long)(rD),         \
      (const uint8_t)(SHIFT))

#define __riscv_cv_mac_machhuRN(rs1, rs2, rD, SHIFT)                           \
  (unsigned long)__builtin_riscv_cv_mac_machhuRN(                              \
      (unsigned long)(rs1), (unsigned long)(rs2), (unsigned long)(rD),         \
      (const uint8_t)(SHIFT))

#define __riscv_cv_mac_macsRN(rs1, rs2, rD, SHIFT)                             \
  (long)__builtin_riscv_cv_mac_macsRN(                                         \
      (unsigned long)(rs1), (unsigned long)(rs2), (unsigned long)(rD),         \
      (const uint8_t)(SHIFT))

#define __riscv_cv_mac_machhsRN(rs1, rs2, rD, SHIFT)                           \
  (long)__builtin_riscv_cv_mac_machhsRN(                                       \
      (unsigned long)(rs1), (unsigned long)(rs2), (unsigned long)(rD),         \
      (const uint8_t)(SHIFT))

#endif // defined(__riscv_xcvmac)

#if defined(__cplusplus)
}
#endif

#endif // define __RISCV_COREV_MAC_H
