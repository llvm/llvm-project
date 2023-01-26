//===-- Single-precision general exp/log functions ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "inv_trigf_utils.h"

namespace __llvm_libc {

// N[Table[ArcTan[x], {x, 1/16, 16/16, 1/16}], 40]
alignas(64) const double ATAN_T[ATAN_T_SIZE] = {
    0x1.ff55bb72cfdeap-5, 0x1.fd5ba9aac2f6ep-4, 0x1.7b97b4bce5b02p-3,
    0x1.f5b75f92c80ddp-3, 0x1.362773707ebccp-2, 0x1.6f61941e4def1p-2,
    0x1.a64eec3cc23fdp-2, 0x1.dac670561bb4fp-2, 0x1.0657e94db30d0p-1,
    0x1.1e00babdefeb4p-1, 0x1.345f01cce37bbp-1, 0x1.4978fa3269ee1p-1,
    0x1.5d58987169b18p-1, 0x1.700a7c5784634p-1, 0x1.819d0b7158a4dp-1,
    0x1.921fb54442d18p-1};

// for(int i = 0; i < 5; i++)
//     printf("%.13a,\n", (-2 * (i % 2) + 1) * 1.0 / (2 * i + 1));
alignas(64) const double ATAN_K[5] = {
    0x1.0000000000000p+0, -0x1.5555555555555p-2, 0x1.999999999999ap-3,
    -0x1.2492492492492p-3, 0x1.c71c71c71c71cp-4};

} // namespace __llvm_libc
