//===- arm_sve_intrinsic_guards.td - SVE builtin guard checks -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TableGen input for checking SVE builtin and LLVM intrinsic target guards.
// Intrinsics.td includes IntrinsicsAArch64.td.
include "llvm/IR/Intrinsics.td"
include "arm_sve.td"
