//===-------lib/Semantics/check-data.h ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_DATA_H_
#define FORTRAN_SEMANTICS_CHECK_DATA_H_

#include "flang/Common/interval.h"
#include "flang/Evaluate/fold-designator.h"
#include "flang/Evaluate/initial-image.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/semantics.h"
#include <list>
#include <map>
#include <vector>

namespace Fortran::parser {
struct DataStmtRepeat;
struct DataStmtObject;
struct DataIDoObject;
class DataStmtImpliedDo;
struct DataStmtSet;
} // namespace Fortran::parser

namespace Fortran::semantics {

struct SymbolDataInitialization {
  using Range = common::Interval<ConstantSubscript>;
  explicit SymbolDataInitialization(std::size_t bytes) : image{bytes} {}
  evaluate::InitialImage image;
  std::list<Range> inits;
};

using DataInitializations = std::map<SymbolRef, SymbolDataInitialization>;

class DataChecker : public virtual BaseChecker {
public:
  explicit DataChecker(SemanticsContext &context) : exprAnalyzer_{context} {}
  void Leave(const parser::DataStmtObject &);
  void Leave(const parser::DataIDoObject &);
  void Enter(const parser::DataImpliedDo &);
  void Leave(const parser::DataImpliedDo &);
  void Leave(const parser::DataStmtSet &);

  // After all DATA statements have been processed, converts their
  // initializations into per-symbol static initializers.
  void CompileDataInitializationsIntoInitializers();

private:
  ConstantSubscript GetRepetitionCount(const parser::DataStmtRepeat &);
  template <typename T> void CheckIfConstantSubscript(const T &);
  void CheckSubscript(const parser::SectionSubscript &);
  bool CheckAllSubscriptsInDataRef(const parser::DataRef &, parser::CharBlock);
  void ConstructInitializer(const Symbol &, SymbolDataInitialization &);

  DataInitializations inits_;
  evaluate::ExpressionAnalyzer exprAnalyzer_;
  bool currentSetHasFatalErrors_{false};
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_DATA_H_
