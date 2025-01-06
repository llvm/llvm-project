//===-- lib/Semantics/check-do-forall.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_DO_FORALL_H_
#define FORTRAN_SEMANTICS_CHECK_DO_FORALL_H_

#include "flang/Common/idioms.h"
#include "flang/Semantics/semantics.h"

namespace Fortran::parser {
struct AssignmentStmt;
struct CallStmt;
struct ConnectSpec;
struct CycleStmt;
struct DoConstruct;
struct ExitStmt;
struct Expr;
struct ForallAssignmentStmt;
struct ForallConstruct;
struct ForallStmt;
struct InquireSpec;
struct IoControlSpec;
struct OutputImpliedDo;
struct StatVariable;
} // namespace Fortran::parser

namespace Fortran::semantics {

// To specify different statement types used in semantic checking.
ENUM_CLASS(StmtType, CYCLE, EXIT)

// Perform semantic checks on DO and FORALL constructs and statements.
class DoForallChecker : public virtual BaseChecker {
public:
  explicit DoForallChecker(SemanticsContext &context) : context_{context} {}
  void Leave(const parser::AssignmentStmt &);
  void Leave(const parser::CallStmt &);
  void Leave(const parser::ConnectSpec &);
  void Enter(const parser::CycleStmt &);
  void Enter(const parser::DoConstruct &);
  void Leave(const parser::DoConstruct &);
  void Enter(const parser::ForallConstruct &);
  void Leave(const parser::ForallConstruct &);
  void Enter(const parser::ForallStmt &);
  void Leave(const parser::ForallStmt &);
  void Leave(const parser::ForallAssignmentStmt &s);
  void Enter(const parser::ExitStmt &);
  void Enter(const parser::Expr &);
  void Leave(const parser::Expr &);
  void Leave(const parser::InquireSpec &);
  void Leave(const parser::IoControlSpec &);
  void Leave(const parser::OutputImpliedDo &);
  void Leave(const parser::StatVariable &);

private:
  SemanticsContext &context_;
  int exprDepth_{0};
  std::list<SemanticsContext::IndexVarKind> nestedWithinConcurrent_;

  void SayBadLeave(
      StmtType, const char *enclosingStmt, const ConstructNode &) const;
  void CheckDoConcurrentExit(StmtType, const ConstructNode &) const;
  void CheckForBadLeave(StmtType, const ConstructNode &) const;
  void CheckNesting(StmtType, const parser::Name *) const;
};
} // namespace Fortran::semantics
#endif
