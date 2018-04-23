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
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace clang {
  class ASTContext;
  class ASTReader;
  class ASTUnit;
  class Decl;
  class FrontendAction;

namespace serialization {
  class ModuleFile;
}

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

/// \param WrappedAction another frontend action to wrap over or null.
std::unique_ptr<FrontendAction>
createIndexingAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
                     IndexingOptions Opts,
                     std::unique_ptr<FrontendAction> WrappedAction);

void indexASTUnit(ASTUnit &Unit, IndexDataConsumer &DataConsumer,
                  IndexingOptions Opts);

void indexTopLevelDecls(ASTContext &Ctx, ArrayRef<const Decl *> Decls,
                        IndexDataConsumer &DataConsumer, IndexingOptions Opts);

void indexModuleFile(serialization::ModuleFile &Mod, ASTReader &Reader,
                     IndexDataConsumer &DataConsumer, IndexingOptions Opts);

} // namespace index
} // namespace clang

#endif
