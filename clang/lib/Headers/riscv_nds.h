/*===---- riscv_nds.h - Andes intrinsics -----------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_NDS_H
#define __RISCV_NDS_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__riscv_xandesbfhcvt)

#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))

static __inline__ float __DEFAULT_FN_ATTRS __riscv_nds_fcvt_s_bf16(__bf16 bf) {
  return __builtin_riscv_nds_fcvt_s_bf16(bf);
}

static __inline__ __bf16 __DEFAULT_FN_ATTRS __riscv_nds_fcvt_bf16_s(float sf) {
  return __builtin_riscv_nds_fcvt_bf16_s(sf);
}

#endif // defined(__riscv_xandesbfhcvt)

#if defined(__cplusplus)
}
#endif

#endif // define __RISCV_NDS_H
