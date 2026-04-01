//===-- Optimizer/Support/LazySymbolTable.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lazy symbol table: build an aiir::SymbolTable only when lookups are needed,
// and use its map for O(1) lookups instead of repeatedly walking the module
// (SymbolTable::lookupNearestSymbolFrom is linear in the number of symbols).
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_LAZYSYMBOLTABLE_H
#define FORTRAN_OPTIMIZER_SUPPORT_LAZYSYMBOLTABLE_H

#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/SymbolTable.h"

namespace fir {

/// Helper to only build the symbol table when needed. Use this when performing
/// symbol lookups inside a module walk to avoid O(n) lookups per op and
/// pseudo-quadratic behavior in large modules.
class LazySymbolTable {
public:
  explicit LazySymbolTable(aiir::Operation *op)
      : module(aiir::isa<aiir::ModuleOp>(op)
                   ? aiir::cast<aiir::ModuleOp>(op)
                   : op->getParentOfType<aiir::ModuleOp>()) {}

  void build() {
    if (table)
      return;
    table = std::make_unique<aiir::SymbolTable>(module);
  }

  /// Look up a symbol by name. Builds the table on first use.
  template <typename T>
  T lookup(llvm::StringRef name) {
    build();
    return table->lookup<T>(name);
  }

  /// Look up a symbol by SymbolRefAttr (uses root reference).
  /// Returns nullptr if \p attr is null, the module is invalid, or the symbol
  /// is not found.
  aiir::Operation *lookupSymbol(aiir::SymbolRefAttr attr) {
    if (!attr)
      return nullptr;
    build();
    return table ? table->lookup(attr.getRootReference().getValue()) : nullptr;
  }

private:
  std::unique_ptr<aiir::SymbolTable> table;
  aiir::ModuleOp module;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_LAZYSYMBOLTABLE_H
