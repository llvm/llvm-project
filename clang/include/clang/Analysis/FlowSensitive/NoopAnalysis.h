//===-- NoopAnalysis.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a NoopAnalysis class that just uses the builtin transfer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_NOOPANALYSIS_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_NOOPANALYSIS_H

#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Analysis/FlowSensitive/TypeErasedDataflowAnalysis.h"
#include <memory>

namespace clang {
namespace dataflow {

class NoopAnalysis : public DataflowAnalysis {
public:
  using DataflowAnalysis::DataflowAnalysis;
  using Lattice = NoopLattice;

  std::unique_ptr<DataflowLattice> initialElement() override {
    return std::make_unique<NoopLattice>();
  }

  void transfer(const CFGElement &E, DataflowLattice &L,
                Environment &Env) override {}
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_NOOPANALYSIS_H
