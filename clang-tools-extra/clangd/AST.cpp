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
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Index/USRGeneration.h"

using namespace llvm;
namespace clang {
namespace clangd {

// Returns true if the complete name of decl \p D is spelled in the source code.
// This is not the case for
//   * symbols formed via macro concatenation, the spelling location will
//     be "<scratch space>"
//   * symbols controlled and defined by a compile command-line option
//     `-DName=foo`, the spelling location will be "<command line>".
bool isSpelledInSourceCode(const Decl *D) {
  const auto &SM = D->getASTContext().getSourceManager();
  auto Loc = D->getLocation();
  // FIXME: Revisit the strategy, the heuristic is limitted when handling
  // macros, we should use the location where the whole definition occurs.
  if (Loc.isMacroID()) {
    std::string PrintLoc = SM.getSpellingLoc(Loc).printToString(SM);
    if (StringRef(PrintLoc).startswith("<scratch") ||
        StringRef(PrintLoc).startswith("<command line>"))
      return false;
  }
  return true;
}

bool isImplementationDetail(const Decl *D) { return !isSpelledInSourceCode(D); }

SourceLocation findNameLoc(const clang::Decl* D) {
  const auto &SM = D->getASTContext().getSourceManager();
  if (!isSpelledInSourceCode(D))
    // Use the expansion location as spelling location is not interesting.
    return SM.getExpansionRange(D->getLocation()).getBegin();
  return SM.getSpellingLoc(D->getLocation());
}

std::string printQualifiedName(const NamedDecl &ND) {
  std::string QName;
  raw_string_ostream OS(QName);
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

std::string printNamespaceScope(const DeclContext &DC) {
  for (const auto *Ctx = &DC; Ctx != nullptr; Ctx = Ctx->getParent())
    if (const auto *NS = dyn_cast<NamespaceDecl>(Ctx))
      if (!NS->isAnonymousNamespace() && !NS->isInlineNamespace())
        return printQualifiedName(*NS) + "::";
  return "";
}

Optional<SymbolID> getSymbolID(const Decl *D) {
  SmallString<128> USR;
  if (index::generateUSRForDecl(D, USR))
    return None;
  return SymbolID(USR);
}

Optional<SymbolID> getSymbolID(const IdentifierInfo &II, const MacroInfo *MI,
                               const SourceManager &SM) {
  if (MI == nullptr)
    return None;
  SmallString<128> USR;
  if (index::generateUSRForMacro(II.getName(), MI->getDefinitionLoc(), SM, USR))
    return None;
  return SymbolID(USR);
}

} // namespace clangd
} // namespace clang
