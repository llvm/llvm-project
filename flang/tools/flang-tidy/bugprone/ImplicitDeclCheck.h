//===--- ImplicitDeclCheck.h - flang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_IMPLICITDECLCHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_IMPLICITDECLCHECK_H

#include "../FlangTidyCheck.h"
#include "../FlangTidyContext.h"
#include "llvm/ADT/StringRef.h"

namespace Fortran::tidy::bugprone {

/// This check verifies that all variables are declared before use.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/implicit-declaration.html
class ImplicitDeclCheck : public virtual FlangTidyCheck {
public:
  ImplicitDeclCheck(llvm::StringRef name, FlangTidyContext *context);
  virtual ~ImplicitDeclCheck() = default;

private:
  void CheckForImplicitDeclarations(semantics::SemanticsContext &,
                                    const semantics::Scope &);
};

} // namespace Fortran::tidy::bugprone

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_IMPLICITDECLCHECK_H
