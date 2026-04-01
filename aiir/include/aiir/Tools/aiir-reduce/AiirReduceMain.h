//===- AiirReduceMain.h - AIIR Reducer driver -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TOOLS_AIIR_REDUCE_AIIRREDUCEMAIN_H
#define AIIR_TOOLS_AIIR_REDUCE_AIIRREDUCEMAIN_H

#include "aiir/Support/LLVM.h"

namespace aiir {

class AIIRContext;

LogicalResult aiirReduceMain(int argc, char **argv, AIIRContext &context);

} // namespace aiir

#endif // AIIR_TOOLS_AIIR_REDUCE_AIIRREDUCEMAIN_H
