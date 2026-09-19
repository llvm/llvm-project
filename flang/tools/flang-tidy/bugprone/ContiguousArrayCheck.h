//===--- ContiguousArrayCheck.h - flang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_CONTIGUOUSARRAYCHECK_H
#define LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_CONTIGUOUSARRAYCHECK_H

#include "../FlangTidyCheck.h"

namespace Fortran::tidy::bugprone {

/// This check verifies that assumed-shape arrays passed to interfaces are
/// contiguous.
///
/// For the user-facing documentation see:
/// https://flang.llvm.org/@PLACEHOLDER@/contiguous-array.html
class ContiguousArrayCheck : public FlangTidyCheck {
public:
  explicit ContiguousArrayCheck(llvm::StringRef name,
                                FlangTidyContext *context);
  virtual ~ContiguousArrayCheck() = default;

private:
  void CheckForContinguousArray(semantics::SemanticsContext &,
                                const semantics::Scope &);
};

} // namespace Fortran::tidy::bugprone

#endif // LLVM_FLANG_TOOLS_FLANG_TIDY_BUGPRONE_CONTIGUOUSARRAYCHECK_H
