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

#define __dbar(/*ui15*/ _1) __builtin_loongarch_dbar((_1))

#ifdef __cplusplus
}
#endif
#endif /* _LOONGARCH_BASE_INTRIN_H */
