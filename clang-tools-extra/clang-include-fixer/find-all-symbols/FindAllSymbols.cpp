//===-- FindAllSymbols.cpp - find all symbols--------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FindAllSymbols.h"
#include "HeaderMapCollector.h"
#include "PathConfig.h"
#include "SymbolInfo.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang {
namespace find_all_symbols {
namespace {

AST_MATCHER(EnumConstantDecl, isInScopedEnum) {
  if (const auto *ED = dyn_cast<EnumDecl>(Node.getDeclContext()))
    return ED->isScoped();
  return false;
}

AST_POLYMORPHIC_MATCHER(isFullySpecialized,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl, VarDecl,
                                                        CXXRecordDecl)) {
  if (Node.getTemplateSpecializationKind() == TSK_ExplicitSpecialization) {
    bool IsPartialSpecialization =
        llvm::isa<VarTemplatePartialSpecializationDecl>(Node) ||
        llvm::isa<ClassTemplatePartialSpecializationDecl>(Node);
    return !IsPartialSpecialization;
  }
  return false;
}

std::vector<SymbolInfo::Context> GetContexts(const NamedDecl *ND) {
  std::vector<SymbolInfo::Context> Contexts;
  for (const auto *Context = ND->getDeclContext(); Context;
       Context = Context->getParent()) {
    if (llvm::isa<TranslationUnitDecl>(Context) ||
        llvm::isa<LinkageSpecDecl>(Context))
      break;

    assert(llvm::isa<NamedDecl>(Context) &&
           "Expect Context to be a NamedDecl");
    if (const auto *NSD = dyn_cast<NamespaceDecl>(Context)) {
      if (!NSD->isInlineNamespace())
        Contexts.emplace_back(SymbolInfo::ContextType::Namespace,
                              NSD->getName().str());
    } else if (const auto *ED = dyn_cast<EnumDecl>(Context)) {
      Contexts.emplace_back(SymbolInfo::ContextType::EnumDecl,
                            ED->getName().str());
    } else {
      const auto *RD = cast<RecordDecl>(Context);
      Contexts.emplace_back(SymbolInfo::ContextType::Record,
                            RD->getName().str());
    }
  }
  return Contexts;
}

std::optional<SymbolInfo>
CreateSymbolInfo(const NamedDecl *ND, const SourceManager &SM,
                 const HeaderMapCollector *Collector) {
  SymbolInfo::SymbolKind Type;
  if (llvm::isa<VarDecl>(ND)) {
    Type = SymbolInfo::SymbolKind::Variable;
  } else if (llvm::isa<FunctionDecl>(ND)) {
    Type = SymbolInfo::SymbolKind::Function;
  } else if (llvm::isa<TypedefNameDecl>(ND)) {
    Type = SymbolInfo::SymbolKind::TypedefName;
  } else if (llvm::isa<EnumConstantDecl>(ND)) {
    Type = SymbolInfo::SymbolKind::EnumConstantDecl;
  } else if (llvm::isa<EnumDecl>(ND)) {
    Type = SymbolInfo::SymbolKind::EnumDecl;
    // Ignore anonymous enum declarations.
    if (ND->getName().empty())
      return std::nullopt;
  } else {
    assert(llvm::isa<RecordDecl>(ND) &&
           "Matched decl must be one of VarDecl, "
           "FunctionDecl, TypedefNameDecl, EnumConstantDecl, "
           "EnumDecl and RecordDecl!");
    // C-style record decl can have empty name, e.g "struct { ... } var;".
    if (ND->getName().empty())
      return std::nullopt;
    Type = SymbolInfo::SymbolKind::Class;
  }

  SourceLocation Loc = SM.getExpansionLoc(ND->getLocation());
  if (!Loc.isValid()) {
    llvm::errs() << "Declaration " << ND->getDeclName() << "("
                 << ND->getDeclKindName()
                 << ") has invalid declaration location.";
    return std::nullopt;
  }

  std::string FilePath = getIncludePath(SM, Loc, Collector);
  if (FilePath.empty())
    return std::nullopt;

  return SymbolInfo(ND->getNameAsString(), Type, FilePath, GetContexts(ND));
}

} // namespace

void FindAllSymbols::registerMatchers(MatchFinder *MatchFinder) {
  // FIXME: Handle specialization.
  auto IsInSpecialization = hasAncestor(
      decl(anyOf(cxxRecordDecl(isExplicitTemplateSpecialization()),
                 functionDecl(isExplicitTemplateSpecialization()))));

  // Matchers for both C and C++.
  // We only match symbols from header files, i.e. not from main files (see
  // function's comment for detailed explanation).
  auto CommonFilter =
      allOf(unless(isImplicit()), unless(isExpansionInMainFile()));

  auto HasNSOrTUCtxMatcher =
      hasDeclContext(anyOf(namespaceDecl(), translationUnitDecl()));

  // We need separate rules for C record types and C++ record types since some
  // template related matchers are inapplicable on C record declarations.
  //
  // Matchers specific to C++ code.
  // All declarations should be in namespace or translation unit.
  auto CCMatcher =
      allOf(HasNSOrTUCtxMatcher, unless(IsInSpecialization),
            unless(ast_matchers::isTemplateInstantiation()),
            unless(isInstantiated()), unless(isFullySpecialized()));

  // Matchers specific to code in extern "C" {...}.
  auto ExternCMatcher = hasDeclContext(linkageSpecDecl());

  // Matchers for variable declarations.
  //
  // In most cases, `ParmVarDecl` is filtered out by hasDeclContext(...)
  // matcher since the declaration context is usually `MethodDecl`. However,
  // this assumption does not hold for parameters of a function pointer
  // parameter.
  // For example, consider a function declaration:
  //        void Func(void (*)(float), int);
  // The float parameter of the function pointer has an empty name, and its
  // declaration context is an anonymous namespace; therefore, it won't be
  // filtered out by our matchers above.
  auto Vars = varDecl(CommonFilter, anyOf(ExternCMatcher, CCMatcher),
                      unless(parmVarDecl()));

  // Matchers for C-style record declarations in extern "C" {...}.
  auto CRecords = recordDecl(CommonFilter, ExternCMatcher, isDefinition());
  // Matchers for C++ record declarations.
  auto CXXRecords = cxxRecordDecl(CommonFilter, CCMatcher, isDefinition());

  // Matchers for function declarations.
  // We want to exclude friend declaration, but the `DeclContext` of a friend
  // function declaration is not the class in which it is declared, so we need
  // to explicitly check if the parent is a `friendDecl`.
  auto Functions = functionDecl(CommonFilter, unless(hasParent(friendDecl())),
                                anyOf(ExternCMatcher, CCMatcher));

  // Matcher for typedef and type alias declarations.
  //
  // typedef and type alias can come from C-style headers and C++ headers.
  // For C-style headers, `DeclContxet` can be either `TranslationUnitDecl`
  // or `LinkageSpecDecl`.
  // For C++ headers, `DeclContext ` can be either `TranslationUnitDecl`
  // or `NamespaceDecl`.
  // With the following context matcher, we can match `typedefNameDecl` from
  // both C-style headers and C++ headers (except for those in classes).
  // "cc_matchers" are not included since template-related matchers are not
  // applicable on `TypedefNameDecl`.
  auto Typedefs =
      typedefNameDecl(CommonFilter, anyOf(HasNSOrTUCtxMatcher,
                                          hasDeclContext(linkageSpecDecl())));

  // Matchers for enum declarations.
  auto Enums = enumDecl(CommonFilter, isDefinition(),
                        anyOf(HasNSOrTUCtxMatcher, ExternCMatcher));

  // Matchers for enum constant declarations.
  // We only match the enum constants in non-scoped enum declarations which are
  // inside toplevel translation unit or a namespace.
  auto EnumConstants = enumConstantDecl(
      CommonFilter, unless(isInScopedEnum()),
      anyOf(hasDeclContext(enumDecl(HasNSOrTUCtxMatcher)), ExternCMatcher));

  // Most of the time we care about all matchable decls, or all types.
  auto Types = namedDecl(anyOf(CRecords, CXXRecords, Enums));
  auto Decls = namedDecl(anyOf(CRecords, CXXRecords, Enums, Typedefs, Vars,
                               EnumConstants, Functions));

  // We want eligible decls bound to "decl"...
  MatchFinder->addMatcher(Decls.bind("decl"), this);

  // ... and all uses of them bound to "use". These have many cases:
  // Uses of values/functions: these generate a declRefExpr.
  MatchFinder->addMatcher(
      declRefExpr(isExpansionInMainFile(), to(Decls.bind("use"))), this);
  // Uses of function templates:
  MatchFinder->addMatcher(
      declRefExpr(isExpansionInMainFile(),
                  to(functionDecl(hasParent(
                      functionTemplateDecl(has(Functions.bind("use"))))))),
      this);

  // Uses of most types: just look at what the typeLoc refers to.
  MatchFinder->addMatcher(
      typeLoc(isExpansionInMainFile(),
              loc(qualType(hasDeclaration(Types.bind("use"))))),
      this);
  // Uses of typedefs: these are often transparent to hasDeclaration, so we need
  // to handle them explicitly.
  MatchFinder->addMatcher(
      typeLoc(isExpansionInMainFile(),
              loc(typedefType(hasDeclaration(Typedefs.bind("use"))))),
      this);
  // Uses of class templates:
  // The typeLoc names the templateSpecializationType. Its declaration is the
  // ClassTemplateDecl, which contains the CXXRecordDecl we want.
  MatchFinder->addMatcher(
      typeLoc(isExpansionInMainFile(),
              loc(templateSpecializationType(hasDeclaration(
                  classTemplateSpecializationDecl(hasSpecializedTemplate(
                      classTemplateDecl(has(CXXRecords.bind("use"))))))))),
      this);
}

void FindAllSymbols::run(const MatchFinder::MatchResult &Result) {
  // Ignore Results in failing TUs.
  if (Result.Context->getDiagnostics().hasErrorOccurred()) {
    return;
  }

  SymbolInfo::Signals Signals;
  const NamedDecl *ND;
  if ((ND = Result.Nodes.getNodeAs<NamedDecl>("use")))
    Signals.Used = 1;
  else if ((ND = Result.Nodes.getNodeAs<NamedDecl>("decl")))
    Signals.Seen = 1;
  else
    assert(false && "Must match a NamedDecl!");

  const SourceManager *SM = Result.SourceManager;
  if (auto Symbol = CreateSymbolInfo(ND, *SM, Collector)) {
    Filename =
        std::string(SM->getFileEntryRefForID(SM->getMainFileID())->getName());
    FileSymbols[*Symbol] += Signals;
  }
}

void FindAllSymbols::onEndOfTranslationUnit() {
  if (Filename != "") {
    Reporter->reportSymbols(Filename, FileSymbols);
    FileSymbols.clear();
    Filename = "";
  }
}

} // namespace find_all_symbols
} // namespace clang
