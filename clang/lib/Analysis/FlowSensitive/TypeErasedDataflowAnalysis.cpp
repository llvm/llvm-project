//===- TypeErasedDataflowAnalysis.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines type-erased base types and functions for building dataflow
//  analyses that run over Control-Flow Graphs (CFGs).
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include "llvm/ADT/Optional.h"

using namespace clang;
using namespace dataflow;

std::vector<llvm::Optional<TypeErasedDataflowAnalysisState>>
runTypeErasedDataflowAnalysis(const CFG &Cfg,
                              TypeErasedDataflowAnalysis &Analysis,
                              const Environment &InitEnv) {
  // FIXME: Consider enforcing that `Cfg` meets the requirements that
  // are specified in the header. This could be done by remembering
  // what options were used to build `Cfg` and asserting on them here.

  // FIXME: Implement work list-based algorithm to compute the fixed
  // point of `Analysis::transform` for every basic block in `Cfg`.
  return {};
}
