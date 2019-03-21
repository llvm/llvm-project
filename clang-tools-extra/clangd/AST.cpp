//===--- AST.cpp - Utility AST functions  -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ScopedPrinter.h"

namespace clang {
namespace clangd {

// Returns true if the complete name of decl \p D is spelled in the source code.
// This is not the case for:
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
    if (llvm::StringRef(PrintLoc).startswith("<scratch") ||
        llvm::StringRef(PrintLoc).startswith("<command line>"))
      return false;
  }
  return true;
}

bool isImplementationDetail(const Decl *D) { return !isSpelledInSourceCode(D); }

SourceLocation findNameLoc(const clang::Decl *D) {
  const auto &SM = D->getASTContext().getSourceManager();
  if (!isSpelledInSourceCode(D))
    // Use the expansion location as spelling location is not interesting.
    return SM.getExpansionRange(D->getLocation()).getBegin();
  return SM.getSpellingLoc(D->getLocation());
}

static llvm::Optional<llvm::ArrayRef<TemplateArgumentLoc>>
getTemplateSpecializationArgLocs(const NamedDecl &ND) {
  if (auto *Func = llvm::dyn_cast<FunctionDecl>(&ND)) {
    if (auto *Args = Func->getTemplateSpecializationArgsAsWritten())
      return Args->arguments();
  } else if (auto *Cls =
                 llvm::dyn_cast<ClassTemplatePartialSpecializationDecl>(&ND)) {
    if (auto *Args = Cls->getTemplateArgsAsWritten())
      return Args->arguments();
  } else if (auto *Var = llvm::dyn_cast<VarTemplateSpecializationDecl>(&ND))
    return Var->getTemplateArgsInfo().arguments();
  // We return None for ClassTemplateSpecializationDecls because it does not
  // contain TemplateArgumentLoc information.
  return llvm::None;
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
  if (auto Args = getTemplateSpecializationArgLocs(ND))
    printTemplateArgumentList(OS, *Args, Policy);
  else if (auto *Cls = llvm::dyn_cast<ClassTemplateSpecializationDecl>(&ND)) {
    if (auto STL = Cls->getTypeAsWritten()
                       ->getTypeLoc()
                       .getAs<TemplateSpecializationTypeLoc>()) {
      llvm::SmallVector<TemplateArgumentLoc, 8> ArgLocs;
      ArgLocs.reserve(STL.getNumArgs());
      for (unsigned I = 0; I < STL.getNumArgs(); ++I)
        ArgLocs.push_back(STL.getArgLoc(I));
      printTemplateArgumentList(OS, ArgLocs, Policy);
    }
  }
  OS.flush();
  assert(!StringRef(QName).startswith("::"));
  return QName;
}

static const TemplateArgumentList *
getTemplateSpecializationArgs(const NamedDecl &ND) {
  if (auto *Func = llvm::dyn_cast<FunctionDecl>(&ND))
    return Func->getTemplateSpecializationArgs();
  if (auto *Cls = llvm::dyn_cast<ClassTemplateSpecializationDecl>(&ND))
    return &Cls->getTemplateInstantiationArgs();
  if (auto *Var = llvm::dyn_cast<VarTemplateSpecializationDecl>(&ND))
    return &Var->getTemplateInstantiationArgs();
  return nullptr;
}

std::string printName(const ASTContext &Ctx, const NamedDecl &ND) {
  std::string Name;
  llvm::raw_string_ostream Out(Name);
  PrintingPolicy PP(Ctx.getLangOpts());
  // Handle 'using namespace'. They all have the same name - <using-directive>.
  if (auto *UD = llvm::dyn_cast<UsingDirectiveDecl>(&ND)) {
    Out << "using namespace ";
    if (auto *Qual = UD->getQualifier())
      Qual->print(Out, PP);
    UD->getNominatedNamespaceAsWritten()->printName(Out);
    return Out.str();
  }
  ND.getDeclName().print(Out, PP);
  if (!Out.str().empty()) {
    // FIXME(ibiryukov): do not show args not explicitly written by the user.
    if (auto *ArgList = getTemplateSpecializationArgs(ND))
      printTemplateArgumentList(Out, ArgList->asArray(), PP);
    return Out.str();
  }
  // The name was empty, so present an anonymous entity.
  if (isa<NamespaceDecl>(ND))
    return "(anonymous namespace)";
  if (auto *Cls = llvm::dyn_cast<RecordDecl>(&ND))
    return ("(anonymous " + Cls->getKindName() + ")").str();
  if (isa<EnumDecl>(ND))
    return "(anonymous enum)";
  return "(anonymous)";
}

std::string printNamespaceScope(const DeclContext &DC) {
  for (const auto *Ctx = &DC; Ctx != nullptr; Ctx = Ctx->getParent())
    if (const auto *NS = dyn_cast<NamespaceDecl>(Ctx))
      if (!NS->isAnonymousNamespace() && !NS->isInlineNamespace())
        return printQualifiedName(*NS) + "::";
  return "";
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
