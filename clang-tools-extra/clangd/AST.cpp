//===--- AST.cpp - Utility AST functions  -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Index/USRGeneration.h"

namespace clang {
namespace clangd {
using namespace llvm;

SourceLocation findNameLoc(const clang::Decl* D) {
  const auto& SM = D->getASTContext().getSourceManager();
  // FIXME: Revisit the strategy, the heuristic is limitted when handling
  // macros, we should use the location where the whole definition occurs.
  SourceLocation SpellingLoc = SM.getSpellingLoc(D->getLocation());
  if (D->getLocation().isMacroID()) {
    std::string PrintLoc = SpellingLoc.printToString(SM);
    if (llvm::StringRef(PrintLoc).startswith("<scratch") ||
        llvm::StringRef(PrintLoc).startswith("<command line>")) {
      // We use the expansion location for the following symbols, as spelling
      // locations of these symbols are not interesting to us:
      //   * symbols formed via macro concatenation, the spelling location will
      //     be "<scratch space>"
      //   * symbols controlled and defined by a compile command-line option
      //     `-DName=foo`, the spelling location will be "<command line>".
      SpellingLoc = SM.getExpansionRange(D->getLocation()).getBegin();
    }
  }
  return SpellingLoc;
}

std::string printQualifiedName(const NamedDecl &ND) {
  std::string QName;
  llvm::raw_string_ostream OS(QName);
  PrintingPolicy Policy(ND.getASTContext().getLangOpts());
  // Note that inline namespaces are treated as transparent scopes. This
  // reflects the way they're most commonly used for lookup. Ideally we'd
  // include them, but at query time it's hard to find all the inline
  // namespaces to query: the preamble doesn't have a dedicated list.
  Policy.SuppressUnwrittenScope = true;
  ND.printQualifiedName(OS, Policy);
  OS.flush();
  assert(!StringRef(QName).startswith("::"));
  return QName;
}

llvm::Optional<SymbolID> getSymbolID(const Decl *D) {
  llvm::SmallString<128> USR;
  if (index::generateUSRForDecl(D, USR))
    return None;
  return SymbolID(USR);
}

llvm::Optional<SymbolID> getSymbolID(const IdentifierInfo &II,
                                     const MacroInfo *MI,
                                     const SourceManager &SM) {
  if (MI == nullptr)
    return None;
  llvm::SmallString<128> USR;
  if (index::generateUSRForMacro(II.getName(), MI->getDefinitionLoc(), SM, USR))
    return None;
  return SymbolID(USR);
}

} // namespace clangd
} // namespace clang
