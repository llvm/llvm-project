//===-- lib/Semantics/data-to-inits.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_DATA_TO_INITS_H_
#define FORTRAN_SEMANTICS_DATA_TO_INITS_H_

#include "flang/Common/interval.h"
#include "flang/Evaluate/fold-designator.h"
#include "flang/Evaluate/initial-image.h"
#include "flang/Support/default-kinds.h"
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
  struct Item {
    Item(Range r, bool isD) : range{r}, isDuplicate{isD} {}
    bool operator<(const Item &that) const { return range < that.range; }
    Range range;
    bool isDuplicate;
  };
  explicit SymbolDataInitialization(std::size_t bytes) : image{bytes} {}
  SymbolDataInitialization(SymbolDataInitialization &&) = default;

  void NoteInitializedRange(Range range, bool isDuplicate = false) {
    if (!initializationItems.empty()) {
      auto &last{initializationItems.back()};
      if (last.isDuplicate == isDuplicate &&
          last.range.AnnexIfPredecessor(range)) {
        return;
      }
    }
    if (!range.empty()) {
      initializationItems.emplace_back(range, isDuplicate);
    }
  }
  void NoteInitializedRange(common::ConstantSubscript offset, std::size_t size,
      bool isDuplicate = false) {
    NoteInitializedRange(Range{offset, size}, isDuplicate);
  }
  void NoteInitializedRange(
      evaluate::OffsetSymbol offsetSymbol, bool isDuplicate = false) {
    NoteInitializedRange(
        offsetSymbol.offset(), offsetSymbol.size(), isDuplicate);
  }

  evaluate::InitialImage image;
  std::list<Item> initializationItems;
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

void ConvertToInitializers(DataInitializations &,
    evaluate::ExpressionAnalyzer &, bool forDerivedTypes = true);

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_DATA_TO_INITS_H_
