//===--- IndexingAction.h - Frontend index action ---------------*- C++ -*-===//
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
#include "clang/Lex/PPCallbacks.h"
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
  bool IndexImplicitInstantiation = false;
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

/// Creates a frontend action that indexes all symbols (macros and AST decls).
/// \param WrappedAction another frontend action to wrap over or null.
std::unique_ptr<FrontendAction>
createIndexingAction(std::shared_ptr<IndexDataConsumer> DataConsumer,
                     IndexingOptions Opts,
                     std::unique_ptr<FrontendAction> WrappedAction);

/// Recursively indexes all decls in the AST.
/// Note that this does not index macros.
void indexASTUnit(ASTUnit &Unit, IndexDataConsumer &DataConsumer,
                  IndexingOptions Opts);

/// Recursively indexes \p Decls.
/// Note that this does not index macros.
void indexTopLevelDecls(ASTContext &Ctx, ArrayRef<const Decl *> Decls,
                        IndexDataConsumer &DataConsumer, IndexingOptions Opts);

/// Creates a PPCallbacks that indexes macros and feeds macros to \p Consumer.
/// The caller is responsible for calling `Consumer.setPreprocessor()`.
std::unique_ptr<PPCallbacks> indexMacrosCallback(IndexDataConsumer &Consumer,
                                                 IndexingOptions Opts);

/// Recursively indexes all top-level decls in the module.
/// FIXME: make this index macros as well.
void indexModuleFile(serialization::ModuleFile &Mod, ASTReader &Reader,
                     IndexDataConsumer &DataConsumer, IndexingOptions Opts);

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
