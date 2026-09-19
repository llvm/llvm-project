//===--- PureProcedureCheck.h - flang-tidy ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_PERFORMANCE_PUREPROCEDURECHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_PERFORMANCE_PUREPROCEDURECHECK_H

#include "../FlangTidyCheck.h"
#include "../FlangTidyContext.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/ADT/StringRef.h"

namespace Fortran::tidy::performance {

/// This check warns about procedures that could be pure but are not.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/pure-procedure.html
class PureProcedureCheck : public virtual FlangTidyCheck {
public:
  explicit PureProcedureCheck(llvm::StringRef name, FlangTidyContext *context);
  virtual ~PureProcedureCheck() = default;
  void SetImpure();

  // I/O
  void Leave(const parser::BackspaceStmt &) override;
  void Leave(const parser::CloseStmt &) override;
  void Leave(const parser::EndfileStmt &) override;
  void Leave(const parser::FlushStmt &) override;
  void Leave(const parser::InquireStmt &) override;
  void Leave(const parser::OpenStmt &) override;
  void Leave(const parser::PrintStmt &) override;
  void Leave(const parser::ReadStmt &) override;
  void Leave(const parser::RewindStmt &) override;
  void Leave(const parser::WaitStmt &) override;
  void Leave(const parser::WriteStmt &) override;

  // Assignment
  void Leave(const parser::AssignmentStmt &) override;

  void Enter(const parser::ExecutableConstruct &) override;
  void Enter(const parser::CallStmt &) override;

  // ?
  void Leave(const parser::Name &) override;

  // Leave the whole program
  void Leave(const parser::Program &) override;

private:
  // map of all procedures and their pure status
  std::unordered_map<const semantics::Symbol *, bool> pureProcedures_;
};

} // namespace Fortran::tidy::performance

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_PERFORMANCE_PUREPROCEDURECHECK_H
