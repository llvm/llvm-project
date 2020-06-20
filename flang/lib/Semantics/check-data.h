//===-------lib/Semantics/check-data.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_DATA_H_
#define FORTRAN_SEMANTICS_CHECK_DATA_H_

#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {
class DataChecker : public virtual BaseChecker {
public:
  explicit DataChecker(SemanticsContext &context) : exprAnalyzer_{context} {}
  void Leave(const parser::DataStmtObject &);
  void Enter(const parser::DataImpliedDo &);
  void Leave(const parser::DataImpliedDo &);
  void Leave(const parser::DataIDoObject &);

private:
  evaluate::ExpressionAnalyzer exprAnalyzer_;
  template <typename T> void CheckIfConstantSubscript(const T &);
  void CheckSubscript(const parser::SectionSubscript &);
  bool CheckAllSubscriptsInDataRef(const parser::DataRef &, parser::CharBlock);
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_DATA_H_
