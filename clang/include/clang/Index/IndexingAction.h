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
#include <string>

namespace clang {
  class ASTContext;
  class ASTReader;
  class ASTUnit;
  class CompilerInstance;
  class Decl;
  class FrontendAction;
  class FrontendOptions;
  class Module;

namespace serialization {
  class ModuleFile;
}

namespace index {
  class IndexDataConsumer;
  class IndexUnitWriter;

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

struct RecordingOptions {
  enum class IncludesRecordingKind {
    None,
    UserOnly, // only record includes inside non-system files.
    All,
  };

  std::string DataDirPath;
  bool RecordSymbolCodeGenName = false;
  bool RecordSystemDependencies = true;
  IncludesRecordingKind RecordIncludes = IncludesRecordingKind::UserOnly;
};

/// \param WrappedAction another frontend action to wrap over or null.
std::unique_ptr<FrontendAction>
createIndexingAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
                     IndexingOptions Opts,
                     std::unique_ptr<FrontendAction> WrappedAction);

void indexASTUnit(ASTUnit &Unit,
                  std::shared_ptr<IndexDataConsumer> DataConsumer,
                  IndexingOptions Opts);

void indexTopLevelDecls(ASTContext &Ctx, ArrayRef<const Decl *> Decls,
                        std::shared_ptr<IndexDataConsumer> DataConsumer,
                        IndexingOptions Opts);

void indexModuleFile(serialization::ModuleFile &Mod, ASTReader &Reader,
                     std::shared_ptr<IndexDataConsumer> DataConsumer,
                     IndexingOptions Opts);

/// \param WrappedAction another frontend action to wrap over or null.
std::unique_ptr<FrontendAction>
createIndexDataRecordingAction(const FrontendOptions &FEOpts,
                               std::unique_ptr<FrontendAction> WrappedAction);

/// Checks if the unit file exists for the module file, if it doesn't it
/// generates index data for it.
///
/// \returns true if the index data were generated, false otherwise.
bool emitIndexDataForModuleFile(const Module *Mod, const CompilerInstance &CI,
                                IndexUnitWriter &ParentUnitWriter);

} // namespace index
} // namespace clang

#endif
