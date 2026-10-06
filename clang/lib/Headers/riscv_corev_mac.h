/*===---- riscv_corev_mac.h - CORE-V multiply-accumulate intrinsics --------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 *
 * This header provides C intrinsics for the CORE-V XCVmac ISA extension.
 * Include this header when compiling with -march=..._xcvmac.
 *
 * Spec: CV32E40P user manual, Instruction Set Extensions (XCVmac):
 *       https://docs.openhwgroup.org/projects/cv32e40p-user-manual/en/latest/
 *       instruction_set_extensions.html
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

#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __artificial__))

/* ---------------------------------------------------------------------------
 * 32x32-bit MAC / MSU
 *
 * cv.mac  rd, rs1, rs2 -> rd += rs1 * rs2
 * cv.msu  rd, rs1, rs2 -> rd -= rs1 * rs2
 *
 * These are the only two operations that take a plain GPR accumulator
 * (no shift/round), so they can be static inline functions.
 * ---------------------------------------------------------------------------
 */

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_mac_mac(int32_t a, int32_t b, int32_t accumulator) {
  return __builtin_riscv_cv_mac_mac(a, b, accumulator);
}

static __inline__ int32_t __DEFAULT_FN_ATTRS
__riscv_cv_mac_msu(int32_t a, int32_t b, int32_t accumulator) {
  return __builtin_riscv_cv_mac_msu(a, b, accumulator);
}

/* ---------------------------------------------------------------------------
 * 16x16-bit multiply (lower halfwords), unsigned, with normalisation shift
 *
 * cv.mulun   rd, rs1, rs2, imm5  -> rd = (u16(rs1) * u16(rs2)) >> imm5
 * cv.mulhhun rd, rs1, rs2, imm5  -> rd = (u16hi(rs1) * u16hi(rs2)) >> imm5
 * cv.mulurn  rd, rs1, rs2, imm5  -> rd = (u16(rs1) * u16(rs2) + rnd) >> imm5
 * cv.mulhhurn rd, rs1, rs2, imm5 -> rd = (u16hi(rs1)*u16hi(rs2) + rnd) >> imm5
 *
 * SHIFT must be a compile-time constant in [0, 31].
 * These are macros so the SHIFT argument is passed as an immediate.
 * ---------------------------------------------------------------------------
 */

#define __riscv_cv_mac_muluN(__rs1, __rs2, __SHIFT)                            \
  ((uint32_t)__builtin_riscv_cv_mac_muluN(                                     \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_mulhhuN(__rs1, __rs2, __SHIFT)                          \
  ((uint32_t)__builtin_riscv_cv_mac_mulhhuN(                                   \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_muluRN(__rs1, __rs2, __SHIFT)                           \
  ((uint32_t)__builtin_riscv_cv_mac_muluRN(                                    \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_mulhhuRN(__rs1, __rs2, __SHIFT)                         \
  ((uint32_t)__builtin_riscv_cv_mac_mulhhuRN(                                  \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__SHIFT)))

/* ---------------------------------------------------------------------------
 * 16x16-bit multiply (lower halfwords), signed, with normalisation shift
 *
 * cv.mulsn    rd, rs1, rs2, imm5 -> rd = (s16(rs1) * s16(rs2)) >> imm5
 * cv.mulhhsn  rd, rs1, rs2, imm5 -> rd = (s16hi(rs1)*s16hi(rs2)) >> imm5
 * cv.mulsrn   rd, rs1, rs2, imm5 -> rd = (s16(rs1) * s16(rs2) + rnd) >> imm5
 * cv.mulhhsrn rd, rs1, rs2, imm5 -> rd = (s16hi(rs1)*s16hi(rs2)+rnd) >> imm5
 *
 * SHIFT must be a compile-time constant in [0, 31].
 * ---------------------------------------------------------------------------
 */

#define __riscv_cv_mac_mulsN(__rs1, __rs2, __SHIFT)                            \
  ((int32_t)__builtin_riscv_cv_mac_mulsN((uint32_t)(__rs1), (uint32_t)(__rs2), \
                                         (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_mulhhsN(__rs1, __rs2, __SHIFT)                          \
  ((int32_t)__builtin_riscv_cv_mac_mulhhsN(                                    \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_mulsRN(__rs1, __rs2, __SHIFT)                           \
  ((int32_t)__builtin_riscv_cv_mac_mulsRN(                                     \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_mulhhsRN(__rs1, __rs2, __SHIFT)                         \
  ((int32_t)__builtin_riscv_cv_mac_mulhhsRN(                                   \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__SHIFT)))

/* ---------------------------------------------------------------------------
 * 16x16-bit multiply-accumulate (lower halfwords), unsigned
 *
 * cv.macun    rd, rs1, rs2, imm5 -> rd = (rd + u16(rs1) * u16(rs2)) >> imm5
 * cv.machhun  rd, rs1, rs2, imm5 -> rd = (rd + u16hi(rs1)*u16hi(rs2)) >> imm5
 * cv.macurn   rd, rs1, rs2, imm5 -> rd = (rd + u16(rs1)*u16(rs2)+rnd) >> imm5
 * cv.machhurn rd, rs1, rs2, imm5 -> rd = (rd+u16hi(rs1)*u16hi(rs2)+rnd)>>imm5
 *
 * SHIFT must be a compile-time constant in [0, 31].
 * rD is the accumulator register (read-modify-write).
 * ---------------------------------------------------------------------------
 */

#define __riscv_cv_mac_macuN(__rD, __rs1, __rs2, __SHIFT)                      \
  ((uint32_t)__builtin_riscv_cv_mac_macuN((uint32_t)(__rs1),                   \
                                          (uint32_t)(__rs2), (uint32_t)(__rD), \
                                          (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_machhuN(__rD, __rs1, __rs2, __SHIFT)                    \
  ((uint32_t)__builtin_riscv_cv_mac_machhuN(                                   \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__rD),                  \
      (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_macuRN(__rD, __rs1, __rs2, __SHIFT)                     \
  ((uint32_t)__builtin_riscv_cv_mac_macuRN(                                    \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__rD),                  \
      (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_machhuRN(__rD, __rs1, __rs2, __SHIFT)                   \
  ((uint32_t)__builtin_riscv_cv_mac_machhuRN(                                  \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__rD),                  \
      (uint32_t)(__SHIFT)))

/* ---------------------------------------------------------------------------
 * 16x16-bit multiply-accumulate (lower halfwords), signed
 *
 * cv.macsn    rd, rs1, rs2, imm5 -> rd = (rd + s16(rs1) * s16(rs2)) >> imm5
 * cv.machhsn  rd, rs1, rs2, imm5 -> rd = (rd+s16hi(rs1)*s16hi(rs2)) >> imm5
 * cv.macsrn   rd, rs1, rs2, imm5 -> rd = (rd+s16(rs1)*s16(rs2)+rnd) >> imm5
 * cv.machhsrn rd, rs1, rs2, imm5 -> rd = (rd+s16hi*s16hi+rnd) >> imm5
 *
 * SHIFT must be a compile-time constant in [0, 31].
 * ---------------------------------------------------------------------------
 */

#define __riscv_cv_mac_macsN(__rD, __rs1, __rs2, __SHIFT)                      \
  ((int32_t)__builtin_riscv_cv_mac_macsN((uint32_t)(__rs1), (uint32_t)(__rs2), \
                                         (uint32_t)(__rD),                     \
                                         (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_machhsN(__rD, __rs1, __rs2, __SHIFT)                    \
  ((int32_t)__builtin_riscv_cv_mac_machhsN(                                    \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__rD),                  \
      (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_macsRN(__rD, __rs1, __rs2, __SHIFT)                     \
  ((int32_t)__builtin_riscv_cv_mac_macsRN((uint32_t)(__rs1),                   \
                                          (uint32_t)(__rs2), (uint32_t)(__rD), \
                                          (uint32_t)(__SHIFT)))

#define __riscv_cv_mac_machhsRN(__rD, __rs1, __rs2, __SHIFT)                   \
  ((int32_t)__builtin_riscv_cv_mac_machhsRN(                                   \
      (uint32_t)(__rs1), (uint32_t)(__rs2), (uint32_t)(__rD),                  \
      (uint32_t)(__SHIFT)))

#endif /* __riscv_xcvmac */

#if defined(__cplusplus)
}
#endif

#endif /* __RISCV_COREV_MAC_H */
