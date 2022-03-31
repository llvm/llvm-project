//===--- SymbolCollector.h ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOLCOLLECTOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOLCOLLECTOR_H

#include "index/CanonicalIncludes.h"
#include "CollectMacros.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "index/Symbol.h"
#include "index/SymbolOrigin.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/ADT/DenseMap.h"
#include <functional>

namespace clang {
namespace clangd {

/// Collect declarations (symbols) from an AST.
/// It collects most declarations except:
/// - Implicit declarations
/// - Anonymous declarations (anonymous enum/class/struct, etc)
/// - Declarations in anonymous namespaces in headers
/// - Local declarations (in function bodies, blocks, etc)
/// - Template specializations
/// - Library-specific private declarations (e.g. private declaration generated
/// by protobuf compiler)
///
/// References to main-file symbols are not collected.
///
/// See also shouldCollectSymbol(...).
///
/// Clients (e.g. clangd) can use SymbolCollector together with
/// index::indexTopLevelDecls to retrieve all symbols when the source file is
/// changed.
class SymbolCollector : public index::IndexDataConsumer {
public:
  struct Options {
    /// When symbol paths cannot be resolved to absolute paths (e.g. files in
    /// VFS that does not have absolute path), combine the fallback directory
    /// with symbols' paths to get absolute paths. This must be an absolute
    /// path.
    std::string FallbackDir;
    bool CollectIncludePath = false;
    /// If set, this is used to map symbol #include path to a potentially
    /// different #include path.
    const CanonicalIncludes *Includes = nullptr;
    // Populate the Symbol.References field.
    bool CountReferences = false;
    /// The symbol ref kinds that will be collected.
    /// If not set, SymbolCollector will not collect refs.
    /// Note that references of namespace decls are not collected, as they
    /// contribute large part of the index, and they are less useful compared
    /// with other decls.
    RefKind RefFilter = RefKind::Unknown;
    /// If set to true, SymbolCollector will collect all refs (from main file
    /// and included headers); otherwise, only refs from main file will be
    /// collected.
    /// This flag is only meaningful when RefFilter is set.
    bool RefsInHeaders = false;
    // Every symbol collected will be stamped with this origin.
    SymbolOrigin Origin = SymbolOrigin::Unknown;
    /// Collect macros.
    /// Note that SymbolCollector must be run with preprocessor in order to
    /// collect macros. For example, `indexTopLevelDecls` will not index any
    /// macro even if this is true.
    bool CollectMacro = false;
    /// Collect symbols local to main-files, such as static functions, symbols
    /// inside an anonymous namespace, function-local classes and its member
    /// functions.
    bool CollectMainFileSymbols = true;
    /// Collect references to main-file symbols.
    bool CollectMainFileRefs = false;
    /// Collect symbols with reserved names, like __Vector_base.
    /// This does not currently affect macros (many like _WIN32 are important!)
    bool CollectReserved = false;
    /// If set to true, SymbolCollector will collect doc for all symbols.
    /// Note that documents of symbols being indexed for completion will always
    /// be collected regardless of this option.
    bool StoreAllDocumentation = false;
    /// If this is set, only collect symbols/references from a file if
    /// `FileFilter(SM, FID)` is true. If not set, all files are indexed.
    std::function<bool(const SourceManager &, FileID)> FileFilter = nullptr;
  };

  SymbolCollector(Options Opts);
  ~SymbolCollector();

  /// Returns true is \p ND should be collected.
  static bool shouldCollectSymbol(const NamedDecl &ND, const ASTContext &ASTCtx,
                                  const Options &Opts, bool IsMainFileSymbol);

  void initialize(ASTContext &Ctx) override;

  void setPreprocessor(std::shared_ptr<Preprocessor> PP) override {
    this->PP = PP.get();
  }
  void setPreprocessor(Preprocessor &PP) { this->PP = &PP; }

  bool
  handleDeclOccurrence(const Decl *D, index::SymbolRoleSet Roles,
                       ArrayRef<index::SymbolRelation> Relations,
                       SourceLocation Loc,
                       index::IndexDataConsumer::ASTNodeInfo ASTNode) override;

  bool handleMacroOccurrence(const IdentifierInfo *Name, const MacroInfo *MI,
                             index::SymbolRoleSet Roles,
                             SourceLocation Loc) override;

  void handleMacros(const MainFileMacros &MacroRefsToIndex);

  SymbolSlab takeSymbols() { return std::move(Symbols).build(); }
  RefSlab takeRefs() { return std::move(Refs).build(); }
  RelationSlab takeRelations() { return std::move(Relations).build(); }

  /// Returns true if we are interested in references and declarations from \p
  /// FID. If this function return false, bodies of functions inside those files
  /// will be skipped to decrease indexing time.
  bool shouldIndexFile(FileID FID);

  void finish() override;

private:
  const Symbol *addDeclaration(const NamedDecl &, SymbolID,
                               bool IsMainFileSymbol);
  void addDefinition(const NamedDecl &, const Symbol &DeclSymbol);
  void processRelations(const NamedDecl &ND, const SymbolID &ID,
                        ArrayRef<index::SymbolRelation> Relations);

  llvm::Optional<SymbolLocation> getTokenLocation(SourceLocation TokLoc);

  llvm::Optional<std::string> getIncludeHeader(const Symbol &S, FileID);

  // All Symbols collected from the AST.
  SymbolSlab::Builder Symbols;
  // File IDs for Symbol.IncludeHeaders.
  // The final spelling is calculated in finish().
  llvm::DenseMap<SymbolID, FileID> IncludeFiles;
  void setIncludeLocation(const Symbol &S, SourceLocation);
  // Indexed macros, to be erased if they turned out to be include guards.
  llvm::DenseSet<const IdentifierInfo *> IndexedMacros;
  // All refs collected from the AST. It includes:
  //   1) symbols declared in the preamble and referenced from the main file (
  //     which is not a header), or
  //   2) symbols declared and referenced from the main file (which is a header)
  RefSlab::Builder Refs;
  // All relations collected from the AST.
  RelationSlab::Builder Relations;
  ASTContext *ASTCtx;
  Preprocessor *PP = nullptr;
  std::shared_ptr<GlobalCodeCompletionAllocator> CompletionAllocator;
  std::unique_ptr<CodeCompletionTUInfo> CompletionTUInfo;
  Options Opts;
  struct SymbolRef {
    SourceLocation Loc;
    index::SymbolRoleSet Roles;
    const Decl *Container;
  };
  // Symbols referenced from the current TU, flushed on finish().
  llvm::DenseSet<const NamedDecl *> ReferencedDecls;
  llvm::DenseSet<const IdentifierInfo *> ReferencedMacros;
  llvm::DenseMap<const NamedDecl *, std::vector<SymbolRef>> DeclRefs;
  llvm::DenseMap<SymbolID, std::vector<SymbolRef>> MacroRefs;
  // Maps canonical declaration provided by clang to canonical declaration for
  // an index symbol, if clangd prefers a different declaration than that
  // provided by clang. For example, friend declaration might be considered
  // canonical by clang but should not be considered canonical in the index
  // unless it's a definition.
  llvm::DenseMap<const Decl *, const Decl *> CanonicalDecls;
  // Cache whether to index a file or not.
  llvm::DenseMap<FileID, bool> FilesToIndexCache;
  // Encapsulates calculations and caches around header paths, which headers
  // to insert for which symbol, etc.
  class HeaderFileURICache;
  std::unique_ptr<HeaderFileURICache> HeaderFileURIs;
};

} // namespace clangd
} // namespace clang

#endif
