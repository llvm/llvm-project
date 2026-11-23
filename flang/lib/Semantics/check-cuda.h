//===-- lib/Semantics/check-cuda.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_CUDA_H_
#define FORTRAN_SEMANTICS_CHECK_CUDA_H_

#include "flang/Semantics/semantics.h"
#include <list>

namespace Fortran::parser {
struct Program;
class Messages;
struct Name;
class CharBlock;
struct AssignmentStmt;
struct ExecutionPartConstruct;
struct ExecutableConstruct;
struct ActionStmt;
struct IfConstruct;
struct CUFKernelDoConstruct;
struct SubroutineSubprogram;
struct FunctionSubprogram;
struct SeparateModuleSubprogram;
} // namespace Fortran::parser

namespace Fortran::semantics {

class SemanticsContext;

class CUDAChecker : public virtual BaseChecker {
public:
  explicit CUDAChecker(SemanticsContext &c) : context_{c} {}
  void Enter(const parser::SubroutineSubprogram &);
  void Enter(const parser::FunctionSubprogram &);
  void Enter(const parser::SeparateModuleSubprogram &);
  void Enter(const parser::CUFKernelDoConstruct &);
  void Leave(const parser::CUFKernelDoConstruct &);
  void Enter(const parser::AssignmentStmt &);
  void Enter(const parser::OpenACCBlockConstruct &);
  void Leave(const parser::OpenACCBlockConstruct &);
  void Enter(const parser::OpenACCCombinedConstruct &);
  void Leave(const parser::OpenACCCombinedConstruct &);
  void Enter(const parser::OpenACCLoopConstruct &);
  void Leave(const parser::OpenACCLoopConstruct &);
  void Enter(const parser::DoConstruct &);
  void Leave(const parser::DoConstruct &);
  void Enter(const parser::PrintStmt &);
  void Enter(const parser::Scalar<parser::Expr> &);
  void Leave(const parser::Scalar<parser::Expr> &);
  void Enter(const parser::ScalarExpr &);
  void Leave(const parser::ScalarExpr &);
  void Enter(const parser::ScalarIntExpr &);
  void Leave(const parser::ScalarIntExpr &);
  void Enter(const parser::ScalarLogicalExpr &);
  void Leave(const parser::ScalarLogicalExpr &);

private:
  template <typename A> void EnterHostScalarExpr(const A &);

  SemanticsContext &context_;
  int deviceConstructDepth_{0};
  int hostScalarExprDepth_{0};
};

bool CanonicalizeCUDA(parser::Program &);

} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_CHECK_CUDA_H_
