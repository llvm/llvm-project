//===--- HeaderSourceSwitch.cpp - --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderSourceSwitch.h"
#include "AST.h"
#include "SourceCode.h"
#include "index/SymbolCollector.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "clang/AST/Decl.h"
#include <optional>

namespace clang {
namespace clangd {

std::optional<Path> getCorrespondingHeaderOrSource(
    PathRef OriginalFile, llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  static constexpr llvm::StringRef SourceExtensions[] = {
      ".cpp", ".c", ".cc", ".cxx", ".c++", ".m", ".mm"};
  static constexpr llvm::StringRef HeaderExtensions[] = {
      ".h",    ".hh",  ".hpp",  ".hxx",  ".inc",
      ".cppm", ".ccm", ".cxxm", ".c++m", ".ixx"};

  llvm::StringRef PathExt = llvm::sys::path::extension(OriginalFile);

  // Lookup in a list of known extensions.
  const bool IsSource =
      llvm::any_of(SourceExtensions, [&PathExt](PathRef SourceExt) {
        return SourceExt.equals_insensitive(PathExt);
      });

  const bool IsHeader =
      llvm::any_of(HeaderExtensions, [&PathExt](PathRef HeaderExt) {
        return HeaderExt.equals_insensitive(PathExt);
      });

  // We can only switch between the known extensions.
  if (!IsSource && !IsHeader)
    return std::nullopt;

  // Array to lookup extensions for the switch. An opposite of where original
  // extension was found.
  llvm::ArrayRef<llvm::StringRef> NewExts;
  if (IsSource)
    NewExts = HeaderExtensions;
  else
    NewExts = SourceExtensions;

  // Storage for the new path.
  llvm::SmallString<128> NewPath = OriginalFile;

  // Loop through switched extension candidates.
  for (llvm::StringRef NewExt : NewExts) {
    llvm::sys::path::replace_extension(NewPath, NewExt);
    if (VFS->exists(NewPath))
      return Path(NewPath);

    // Also check NewExt in upper-case, just in case.
    llvm::sys::path::replace_extension(NewPath, NewExt.upper());
    if (VFS->exists(NewPath))
      return Path(NewPath);
  }
  return std::nullopt;
}

std::optional<Path> getCorrespondingHeaderOrSource(PathRef OriginalFile,
                                                   ParsedAST &AST,
                                                   const SymbolIndex *Index) {
  if (!Index) {
    // FIXME: use the AST to do the inference.
    return std::nullopt;
  }
  LookupRequest Request;
  // Find all symbols present in the original file.
  for (const auto *D : getIndexableLocalDecls(AST)) {
    if (auto ID = getSymbolID(D))
      Request.IDs.insert(ID);
  }
  llvm::StringMap<int> Candidates; // Target path => score.
  // When in the implementation file, we always search for the header file,
  // using the decl loc. When we are in a header, this usually implies searching
  // for implementation, for which we use the definition loc. For templates, we
  // can have separate implementation headers, which behave as an implementation
  // file. As such, we always have to add the decl loc and conditionally
  // definition loc.
  //
  // For each symbol in the original file, we get its target location (decl or
  // def) from the index, then award that target file.
#ifdef CLANGD_PATH_CASE_INSENSITIVE
  auto pathEqual = [](llvm::StringRef l, llvm::StringRef r) {
    return l.equals_insensitive(r);
  };
#else
  auto pathEqual = [](llvm::StringRef l, llvm::StringRef r) { return l == r; };
#endif
  Index->lookup(Request, [&](const Symbol &Sym) {
    if (llvm::StringRef{Sym.Definition.FileURI}.empty() ||
        llvm::StringRef{Sym.CanonicalDeclaration.FileURI}.empty())
      return;
    auto TargetPathDefinition =
        URI::resolve(Sym.Definition.FileURI, OriginalFile);
    if (!TargetPathDefinition)
      return;
    auto TargetPathDeclaration =
        URI::resolve(Sym.CanonicalDeclaration.FileURI, OriginalFile);
    if (!TargetPathDeclaration)
      return;
    if (pathEqual(*TargetPathDefinition, OriginalFile)) {
      if (!pathEqual(*TargetPathDeclaration, OriginalFile))
        ++Candidates[*TargetPathDeclaration];
    } else if (pathEqual(*TargetPathDeclaration, OriginalFile))
      ++Candidates[*TargetPathDefinition];
  });
  // FIXME: our index doesn't have any interesting information (this could be
  // that the background-index is not finished), we should use the decl/def
  // locations from the AST to do the inference (from .cc to .h).
  if (Candidates.empty())
    return std::nullopt;

  // Pickup the winner, who contains most of symbols.
  // FIXME: should we use other signals (file proximity) to help score?
  auto Best = Candidates.begin();
  for (auto It = Candidates.begin(); It != Candidates.end(); ++It) {
    if (It->second > Best->second)
      Best = It;
    else if (It->second == Best->second && It->first() < Best->first())
      // Select the first one in the lexical order if we have multiple
      // candidates.
      Best = It;
  }
  return Path(Best->first());
}

std::vector<const Decl *> getIndexableLocalDecls(ParsedAST &AST) {
  std::vector<const Decl *> Results;
  std::function<void(Decl *)> TraverseDecl = [&](Decl *D) {
    auto *ND = llvm::dyn_cast<NamedDecl>(D);
    if (!ND || ND->isImplicit())
      return;
    if (!SymbolCollector::shouldCollectSymbol(*ND, D->getASTContext(), {},
                                              /*IsMainFileSymbol=*/false))
      return;
    if (!llvm::isa<FunctionDecl>(ND)) {
      // Visit the children, but we skip function decls as we are not interested
      // in the function body.
      if (auto *Scope = llvm::dyn_cast<DeclContext>(ND)) {
        for (auto *D : Scope->decls())
          TraverseDecl(D);
      }
      // ClassTemplateDecl does not inherit from DeclContext
      if (auto *Scope = llvm::dyn_cast<ClassTemplateDecl>(ND)) {
        for (auto *D : Scope->getTemplatedDecl()->decls())
          TraverseDecl(D);
      }
    }
    if (llvm::isa<NamespaceDecl>(D))
      return; // namespace is indexable, but we're not interested.
    Results.push_back(D);
  };
  // Traverses the ParsedAST directly to collect all decls present in the main
  // file.
  for (auto *TopLevel : AST.getLocalTopLevelDecls())
    TraverseDecl(TopLevel);
  return Results;
}

} // namespace clangd
} // namespace clang
