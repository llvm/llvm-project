//===--- AccumulatorRaceCheck.h - flang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_OPENMP_ACCUMULATORRACECHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_OPENMP_ACCUMULATORRACECHECK_H
#include "../FlangTidyCheck.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::tidy::openmp {

/// This check tries to detect potential race conditions in OpenMP accumulators.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/accumulator-race.html
class AccumulatorRaceCheck : public virtual FlangTidyCheck {
public:
  using FlangTidyCheck::FlangTidyCheck;
  virtual ~AccumulatorRaceCheck() = default;

  void Enter(const parser::OpenMPBlockConstruct &) override;
  void Leave(const parser::OpenMPBlockConstruct &) override;
  void Enter(const parser::OpenMPLoopConstruct &) override;
  void Leave(const parser::OpenMPLoopConstruct &) override;
  void Enter(const parser::OmpAtomicUpdate &) override;
  void Leave(const parser::OmpAtomicUpdate &) override;
  void Enter(const parser::OpenMPCriticalConstruct &) override;
  void Leave(const parser::OpenMPCriticalConstruct &) override;
  void Enter(const parser::AssignmentStmt &) override;

private:
  int inParallelRegion_ = 0;
  int inCriticalSection_ = 0;
  int inAtomicUpdate_ = 0;
};

} // namespace Fortran::tidy::openmp
#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_OPENMP_ACCUMULATORRACECHECK_H
