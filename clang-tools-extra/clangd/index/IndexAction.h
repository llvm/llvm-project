//===--- IndexAction.h - Run the indexer as a frontend action ----*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_ACTION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEX_ACTION_H
#include "SymbolCollector.h"
#include "clang/Frontend/FrontendActions.h"

namespace clang {
namespace clangd {

// Creates an action that indexes translation units and delivers the results
// for SymbolsCallback (each slab corresponds to one TU).
//
// Only a subset of SymbolCollector::Options are respected:
//   - include paths are always collected, and canonicalized appropriately
//   - references are always counted
//   - the symbol origin is always Static
std::unique_ptr<FrontendAction>
createStaticIndexingAction(SymbolCollector::Options Opts,
                           std::function<void(SymbolSlab)> SymbolsCallback);

} // namespace clangd
} // namespace clang

#endif
