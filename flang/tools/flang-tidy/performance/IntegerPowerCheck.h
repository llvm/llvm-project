//===--- IntegerPowerCheck.h - flang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_PERFORMANCE_INTEGERPOWERCHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_PERFORMANCE_INTEGERPOWERCHECK_H

#include "../FlangTidyCheck.h"
#include "../FlangTidyContext.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/ADT/StringRef.h"

namespace Fortran::tidy::performance {

/// This check warns about procedures that could be pure but are not.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/pure-procedure.html
class IntegerPowerCheck : public virtual FlangTidyCheck {
public:
  explicit IntegerPowerCheck(llvm::StringRef name, FlangTidyContext *context);
  virtual ~IntegerPowerCheck() = default;

  void Enter(const parser::Expr::Power &) override;
};

} // namespace Fortran::tidy::performance

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_PERFORMANCE_INTEGERPOWERCHECK_H
