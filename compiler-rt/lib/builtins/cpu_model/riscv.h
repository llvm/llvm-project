//===-- cpu_model/riscv.h ----------------------------------------------- -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cpu_model.h"

#if !defined(__riscv)
#error This file is intended only for riscv-based targets
#endif

void __init_riscv_feature_bits(void *);
