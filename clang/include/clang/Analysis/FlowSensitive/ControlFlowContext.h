//===-- ControlFlowContext.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a deprecated alias for AdornedCFG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_CONTROLFLOWCONTEXT_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_CONTROLFLOWCONTEXT_H

#include "clang/Analysis/FlowSensitive/AdornedCFG.h"

namespace clang {
namespace dataflow {

// This is a deprecated alias. Use `AdornedCFG` instead.
using ControlFlowContext = AdornedCFG;

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_CONTROLFLOWCONTEXT_H
