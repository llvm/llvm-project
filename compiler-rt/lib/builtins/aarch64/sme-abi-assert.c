// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// We rely on the FMV __aarch64_cpu_features mechanism to determine
// which features are set at runtime.

#include "../cpu_model/AArch64CPUFeatures.inc"
_Static_assert(FEAT_SVE == 30, "sme-abi.S assumes FEAT_SVE = 30");
_Static_assert(FEAT_SME == 42, "sme-abi.S assumes FEAT_SME = 42");
_Static_assert(FEAT_SME2 == 57, "sme-abi.S assumes FEAT_SME2 = 57");
