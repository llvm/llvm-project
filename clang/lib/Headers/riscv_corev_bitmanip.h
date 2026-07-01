/*===---- riscv_corev_bitmanip.h - CORE-V bit-manipulation intrinsics ------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 *
 * This header provides C intrinsics for the CORE-V XCVbitmanip ISA extension.
 * Include this header when compiling with -march=..._xcvbitmanip.
 *
 * Spec: CV32E40P user manual, Instruction Set Extensions (XCVbitmanip):
 *       https://docs.openhwgroup.org/projects/cv32e40p-user-manual/en/latest/
 *       instruction_set_extensions.html
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_COREV_BITMANIP_H
#define __RISCV_COREV_BITMANIP_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__riscv_xcvbitmanip)

#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __artificial__))

/* ---------------------------------------------------------------------------
 * cv.extract rd, rs1, IS3, IS2
 *   rd = SignExtend(rs1[IS3+IS2-1 : IS2], 32)
 *   IS3 = bit-width (uimm5), IS2 = lower bit index (uimm5)
 *   The two immediates are packed as a single uimm10: IS3<<5 | IS2
 *   IMM must be a compile-time constant.
 * ---------------------------------------------------------------------------
 */
#define __riscv_cv_bitmanip_extract(__rs1, __IMM)                              \
  ((int32_t)__builtin_riscv_cv_bitmanip_extract((uint32_t)(__rs1),             \
                                                (uint32_t)(__IMM)))

/* ---------------------------------------------------------------------------
 * cv.extractu rd, rs1, IS3, IS2
 *   rd = ZeroExtend(rs1[IS3+IS2-1 : IS2], 32)
 * ---------------------------------------------------------------------------
 */
#define __riscv_cv_bitmanip_extractu(__rs1, __IMM)                             \
  ((uint32_t)__builtin_riscv_cv_bitmanip_extractu((uint32_t)(__rs1),           \
                                                  (uint32_t)(__IMM)))

/* ---------------------------------------------------------------------------
 * cv.bclr rd, rs1, IS3, IS2
 *   rd = rs1 & ~(((1 << IS3) - 1) << IS2)   (clear IS3 bits starting at IS2)
 * ---------------------------------------------------------------------------
 */
#define __riscv_cv_bitmanip_bclr(__rs1, __IMM)                                 \
  ((uint32_t)__builtin_riscv_cv_bitmanip_bclr((uint32_t)(__rs1),               \
                                              (uint32_t)(__IMM)))

/* ---------------------------------------------------------------------------
 * cv.bset rd, rs1, IS3, IS2
 *   rd = rs1 | (((1 << IS3) - 1) << IS2)    (set IS3 bits starting at IS2)
 * ---------------------------------------------------------------------------
 */
#define __riscv_cv_bitmanip_bset(__rs1, __IMM)                                 \
  ((uint32_t)__builtin_riscv_cv_bitmanip_bset((uint32_t)(__rs1),               \
                                              (uint32_t)(__IMM)))

/* ---------------------------------------------------------------------------
 * cv.insert rd, rs1, IS3, IS2
 *   rd[IS3+IS2-1 : IS2] = rs1[IS3-1 : 0]
 *   rd is read-modify-write (the bits outside the target range are preserved).
 *   IS3 = bit-width (uimm5), IS2 = lower bit index (uimm5)
 *   The two immediates are packed as a single uimm10: IS3<<5 | IS2
 * ---------------------------------------------------------------------------
 */
#define __riscv_cv_bitmanip_insert(__rD, __rs1, __IMM)                         \
  ((uint32_t)__builtin_riscv_cv_bitmanip_insert(                               \
      (uint32_t)(__rs1), (uint32_t)(__IMM), (uint32_t)(__rD)))

/* ---------------------------------------------------------------------------
 * cv.clb rd, rs1
 *   rd = number of consecutive leading bits equal to the sign bit (after the
 *        sign bit itself), i.e. count leading bits equal to bit 31.
 *   Equivalent to max(0, clz(rs1) + clz(~rs1) - 1).
 *
 * Note: ff1 (find-first-one from LSB), fl1 (find-last-one from LSB), cnt
 * (popcount) and ror (rotate-right) are intentionally absent from this header.
 * The compiler automatically selects cv.ff1 / cv.fl1 / cv.cnt / cv.ror from
 * standard C expressions (__builtin_ctz, __builtin_clz, __builtin_popcount,
 * rotate patterns) when -march includes xcvbitmanip. No explicit builtin call
 * is required or provided for those four operations.
 * ---------------------------------------------------------------------------
 */
static __inline__ uint32_t __DEFAULT_FN_ATTRS
__riscv_cv_bitmanip_clb(uint32_t a) {
  return __builtin_riscv_cv_bitmanip_clb(a);
}

/* ---------------------------------------------------------------------------
 * cv.bitrev rd, rs1, IS2, IS3
 *   Reverses groups of (2^IS2) bits within each (IS3+1)-bit segment of rs1.
 *   IS2 = radix (uimm2, 0..3), IS3 = number of significant bits minus 1 (uimm5)
 *   Both IS2 and IS3 must be compile-time constants.
 *
 * Example: cv.bitrev t0, t0, 2, 23 reverses groups of 4 bits within each
 * 24-bit window.
 * ---------------------------------------------------------------------------
 */
#define __riscv_cv_bitmanip_bitrev(__rs1, __IS2, __IS3)                        \
  ((uint32_t)__builtin_riscv_cv_bitmanip_bitrev(                               \
      (uint32_t)(__rs1), (uint32_t)(__IS2), (uint32_t)(__IS3)))

#endif /* __riscv_xcvbitmanip */

#if defined(__cplusplus)
}
#endif

#endif /* __RISCV_COREV_BITMANIP_H */
