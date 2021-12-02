//===--- IndexAction.h - Run the indexer as a frontend action ----*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEXACTION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_INDEXACTION_H
#include "Headers.h"
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
//   - all references are collected (if RefsCallback is non-null)
//   - the symbol origin is set to Static if not specified by caller
std::unique_ptr<FrontendAction> createStaticIndexingAction(
    SymbolCollector::Options Opts,
    std::function<void(SymbolSlab)> SymbolsCallback,
    std::function<void(RefSlab)> RefsCallback,
    std::function<void(RelationSlab)> RelationsCallback,
    std::function<void(IncludeGraph)> IncludeGraphCallback);

} // namespace clangd
} // namespace clang

#endif
