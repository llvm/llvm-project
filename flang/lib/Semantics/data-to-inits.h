//===-- lib/Semantics/data-to-inits.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_DATA_TO_INITS_H_
#define FORTRAN_SEMANTICS_DATA_TO_INITS_H_

#include "flang/Common/default-kinds.h"
#include "flang/Common/interval.h"
#include "flang/Evaluate/fold-designator.h"
#include "flang/Evaluate/initial-image.h"
#include <list>
#include <map>

namespace Fortran::parser {
struct DataStmtSet;
struct DataStmtValue;
} // namespace Fortran::parser
namespace Fortran::evaluate {
class ExpressionAnalyzer;
}
namespace Fortran::semantics {

class Symbol;

struct SymbolDataInitialization {
  using Range = common::Interval<common::ConstantSubscript>;
  explicit SymbolDataInitialization(std::size_t bytes) : image{bytes} {}
  SymbolDataInitialization(SymbolDataInitialization &&) = default;

  void NoteInitializedRange(Range range) {
    if (initializedRanges.empty() ||
        !initializedRanges.back().AnnexIfPredecessor(range)) {
      initializedRanges.emplace_back(range);
    }
  }
  void NoteInitializedRange(
      common::ConstantSubscript offset, std::size_t size) {
    NoteInitializedRange(Range{offset, size});
  }
  void NoteInitializedRange(evaluate::OffsetSymbol offsetSymbol) {
    NoteInitializedRange(offsetSymbol.offset(), offsetSymbol.size());
  }

  evaluate::InitialImage image;
  std::list<Range> initializedRanges;
};

using DataInitializations = std::map<const Symbol *, SymbolDataInitialization>;

// Matches DATA statement variables with their values and checks
// compatibility.
void AccumulateDataInitializations(DataInitializations &,
    evaluate::ExpressionAnalyzer &, const parser::DataStmtSet &);

// For legacy DATA-style initialization extension: integer n(2)/1,2/
void AccumulateDataInitializations(DataInitializations &,
    evaluate::ExpressionAnalyzer &, const Symbol &,
    const std::list<common::Indirection<parser::DataStmtValue>> &);

void ConvertToInitializers(
    DataInitializations &, evaluate::ExpressionAnalyzer &);

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_DATA_TO_INITS_H_
