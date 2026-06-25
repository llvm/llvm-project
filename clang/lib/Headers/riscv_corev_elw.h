/*===---- riscv_corev_elw.h - CORE-V event-load-word intrinsics ------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 *
 * This header provides C intrinsics for the CORE-V XCVelw ISA extension.
 * Include this header when compiling with -march=..._xcvelw.
 *
 * Spec: CV32E40P user manual, Instruction Set Extensions (XCVelw):
 *       https://docs.openhwgroup.org/projects/cv32e40p-user-manual/en/latest/
 *       instruction_set_extensions.html
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_COREV_ELW_H
#define __RISCV_COREV_ELW_H
#include <stdint.h>
#if defined(__cplusplus)
extern "C" {
#endif
#if defined(__riscv_xcvelw)
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __artificial__))

// cv.elw rd, imm(rs1): event load word
static __inline__ int32_t __DEFAULT_FN_ATTRS __riscv_cv_elw_elw(int32_t *ptr) {
  return __builtin_riscv_cv_elw_elw(ptr);
}
#endif
#if defined(__cplusplus)
}
#endif
#endif
