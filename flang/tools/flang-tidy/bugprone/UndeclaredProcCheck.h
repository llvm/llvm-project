//===--- UndeclaredProcCheck.h - flang-tidy ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_UNDECLAREDPROCCHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_UNDECLAREDPROCCHECK_H

#include "../FlangTidyCheck.h"
#include "../FlangTidyContext.h"

namespace Fortran::tidy::bugprone {

/// This check verifies that all procedures are explicitly declared.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/implicit-procedure-declaration.html
class UndeclaredProcCheck : public virtual FlangTidyCheck {
public:
  UndeclaredProcCheck(llvm::StringRef name, FlangTidyContext *context);
  virtual ~UndeclaredProcCheck() = default;

private:
  void CheckForUndeclaredProcedures(semantics::SemanticsContext &,
                                    const semantics::Scope &);
};

} // namespace Fortran::tidy::bugprone

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_UNDECLAREDPROCCHECK_H
