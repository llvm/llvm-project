//===--- IndexingAction.h - Frontend index action -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_INDEXINGACTION_H
#define LLVM_CLANG_INDEX_INDEXINGACTION_H

#include "clang/Basic/LLVM.h"
#include <memory>

namespace clang {
  class ASTUnit;
  class FrontendAction;

namespace index {
  class IndexDataConsumer;

struct IndexingOptions {
  enum class SystemSymbolFilterKind {
    None,
    DeclarationsOnly,
    All,
  };

  SystemSymbolFilterKind SystemSymbolFilter
    = SystemSymbolFilterKind::DeclarationsOnly;
  bool IndexFunctionLocals = false;
};

std::unique_ptr<FrontendAction>
createIndexingAction(std::unique_ptr<FrontendAction> WrappedAction,
                     std::shared_ptr<IndexDataConsumer> DataConsumer,
                     IndexingOptions Opts);

void indexASTUnit(ASTUnit &Unit,
                  std::shared_ptr<IndexDataConsumer> DataConsumer,
                  IndexingOptions Opts);

} // namespace index
} // namespace clang

#endif
