/*===---------- amxfp8intrin.h - AMX intrinsics -*- C++ -*------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===------------------------------------------------------------------------===
 */

#ifndef __IMMINTRIN_H
#error "Never use <amxfp8intrin.h> directly; include <immintrin.h> instead."
#endif /* __IMMINTRIN_H */

#ifndef __AMXFP8INTRIN_H
#define __AMXFP8INTRIN_H
#ifdef __x86_64__

#define _tile_dpbf8ps __builtin_ia32_tdpbf8ps
#define _tile_dpbhf8ps __builtin_ia32_tdpbhf8ps
#define _tile_dphbf8ps __builtin_ia32_tdphbf8ps
#define _tile_dphf8ps __builtin_ia32_tdphf8ps

#endif /* __x86_64__ */
#endif /* __AMXFP8INTRIN_H */
