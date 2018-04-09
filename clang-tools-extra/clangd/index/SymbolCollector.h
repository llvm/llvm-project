//===--- SymbolCollector.h ---------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CanonicalIncludes.h"
#include "Index.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Sema/CodeCompleteConsumer.h"

namespace clang {
namespace clangd {

/// \brief Collect top-level symbols from an AST. These are symbols defined
/// immediately inside a namespace or a translation unit scope. For example,
/// symbols in classes or functions are not collected. Note that this only
/// collects symbols that declared in at least one file that is not a main
/// file (i.e. the source file corresponding to a TU). These are symbols that
/// can be imported by other files by including the file where symbols are
/// declared.
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
    /// Specifies URI schemes that can be used to generate URIs for file paths
    /// in symbols. The list of schemes will be tried in order until a working
    /// scheme is found. If no scheme works, symbol location will be dropped.
    std::vector<std::string> URISchemes = {"file"};
    bool CollectIncludePath = false;
    /// If set, this is used to map symbol #include path to a potentially
    /// different #include path.
    const CanonicalIncludes *Includes = nullptr;
    // Populate the Symbol.References field.
    bool CountReferences = false;
  };

  SymbolCollector(Options Opts);

  void initialize(ASTContext &Ctx) override;

  void setPreprocessor(std::shared_ptr<Preprocessor> PP) override {
    this->PP = std::move(PP);
  }

  bool
  handleDeclOccurence(const Decl *D, index::SymbolRoleSet Roles,
                      ArrayRef<index::SymbolRelation> Relations,
                      SourceLocation Loc,
                      index::IndexDataConsumer::ASTNodeInfo ASTNode) override;

  SymbolSlab takeSymbols() { return std::move(Symbols).build(); }

  void finish() override;

private:
  const Symbol *addDeclaration(const NamedDecl &, SymbolID);
  void addDefinition(const NamedDecl &, const Symbol &DeclSymbol);

  // All Symbols collected from the AST.
  SymbolSlab::Builder Symbols;
  ASTContext *ASTCtx;
  std::shared_ptr<Preprocessor> PP;
  std::shared_ptr<GlobalCodeCompletionAllocator> CompletionAllocator;
  std::unique_ptr<CodeCompletionTUInfo> CompletionTUInfo;
  Options Opts;
  // Decls referenced from the current TU, flushed on finish().
  llvm::DenseSet<const NamedDecl *> ReferencedDecls;
};

} // namespace clangd
} // namespace clang
