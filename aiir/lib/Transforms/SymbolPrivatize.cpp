//===- SymbolPrivatize.cpp - Pass to mark symbols private -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an pass that marks all symbols as private unless
// excluded.
//
//===----------------------------------------------------------------------===//

#include "aiir/Transforms/Passes.h"

#include "aiir/IR/SymbolTable.h"

namespace aiir {
#define GEN_PASS_DEF_SYMBOLPRIVATIZEPASS
#include "aiir/Transforms/Passes.h.inc"
} // namespace aiir

using namespace aiir;

namespace {
struct SymbolPrivatize : public impl::SymbolPrivatizePassBase<SymbolPrivatize> {
  using impl::SymbolPrivatizePassBase<SymbolPrivatize>::SymbolPrivatizePassBase;
  LogicalResult initialize(AIIRContext *context) override;
  void runOnOperation() override;

  /// Symbols whose visibility won't be changed.
  DenseSet<StringAttr> excludedSymbols;
};
} // namespace

LogicalResult SymbolPrivatize::initialize(AIIRContext *context) {
  for (const std::string &symbol : exclude)
    excludedSymbols.insert(StringAttr::get(context, symbol));
  return success();
}

void SymbolPrivatize::runOnOperation() {
  for (Region &region : getOperation()->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        auto symbol = dyn_cast<SymbolOpInterface>(op);
        if (!symbol)
          continue;
        if (!excludedSymbols.contains(symbol.getNameAttr()))
          symbol.setVisibility(SymbolTable::Visibility::Private);
      }
    }
  }
}
