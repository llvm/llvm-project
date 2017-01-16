//===-- FindAllSymbols.cpp - find all symbols--------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FindAllSymbols.h"
#include "HeaderMapCollector.h"
#include "PathConfig.h"
#include "SymbolInfo.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/FileSystem.h"

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

llvm::Optional<SymbolInfo>
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
      return llvm::None;
  } else {
    assert(llvm::isa<RecordDecl>(ND) &&
           "Matched decl must be one of VarDecl, "
           "FunctionDecl, TypedefNameDecl, EnumConstantDecl, "
           "EnumDecl and RecordDecl!");
    // C-style record decl can have empty name, e.g "struct { ... } var;".
    if (ND->getName().empty())
      return llvm::None;
    Type = SymbolInfo::SymbolKind::Class;
  }

  SourceLocation Loc = SM.getExpansionLoc(ND->getLocation());
  if (!Loc.isValid()) {
    llvm::errs() << "Declaration " << ND->getNameAsString() << "("
                 << ND->getDeclKindName()
                 << ") has invalid declaration location.";
    return llvm::None;
  }

  std::string FilePath = getIncludePath(SM, Loc, Collector);
  if (FilePath.empty()) return llvm::None;

  return SymbolInfo(ND->getNameAsString(), Type, FilePath,
                    SM.getExpansionLineNumber(Loc), GetContexts(ND));
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

  // We need seperate rules for C record types and C++ record types since some
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
  MatchFinder->addMatcher(varDecl(CommonFilter,
                                  anyOf(ExternCMatcher, CCMatcher),
                                  unless(parmVarDecl()))
                              .bind("decl"),
                          this);

  // Matchers for C-style record declarations in extern "C" {...}.
  MatchFinder->addMatcher(
      recordDecl(CommonFilter, ExternCMatcher, isDefinition()).bind("decl"),
      this);

  // Matchers for C++ record declarations.
  auto CxxRecordDecl =
      cxxRecordDecl(CommonFilter, CCMatcher, isDefinition());
  MatchFinder->addMatcher(CxxRecordDecl.bind("decl"), this);

  // Matchers for function declarations.
  // We want to exclude friend declaration, but the `DeclContext` of a friend
  // function declaration is not the class in which it is declared, so we need
  // to explicitly check if the parent is a `friendDecl`.
  MatchFinder->addMatcher(functionDecl(CommonFilter,
                                       unless(hasParent(friendDecl())),
                                       anyOf(ExternCMatcher, CCMatcher))
                              .bind("decl"),
                          this);

  // Matcher for typedef and type alias declarations.
  //
  // typedef and type alias can come from C-style headers and C++ heaeders.
  // For C-style header, `DeclContxet` can be either `TranslationUnitDecl`
  // or `LinkageSpecDecl`.
  // For C++ header, `DeclContext ` can be one of `TranslationUnitDecl`,
  // `NamespaceDecl`.
  // With the following context matcher, we can match `typedefNameDecl` from
  // both C-style header and C++ header (except for those in classes).
  // "cc_matchers" are not included since template-related matchers are not
  // applicable on `TypedefNameDecl`.
  MatchFinder->addMatcher(
      typedefNameDecl(CommonFilter, anyOf(HasNSOrTUCtxMatcher,
                                          hasDeclContext(linkageSpecDecl())))
          .bind("decl"),
      this);

  // Matchers for enum declarations.
  MatchFinder->addMatcher(enumDecl(CommonFilter, isDefinition(),
                                   anyOf(HasNSOrTUCtxMatcher, ExternCMatcher))
                              .bind("decl"),
                          this);

  // Matchers for enum constant declarations.
  // We only match the enum constants in non-scoped enum declarations which are
  // inside toplevel translation unit or a namespace.
  MatchFinder->addMatcher(
      enumConstantDecl(
          CommonFilter,
          unless(isInScopedEnum()),
          anyOf(hasDeclContext(enumDecl(HasNSOrTUCtxMatcher)), ExternCMatcher))
          .bind("decl"),
      this);
}

void FindAllSymbols::run(const MatchFinder::MatchResult &Result) {
  // Ignore Results in failing TUs.
  if (Result.Context->getDiagnostics().hasErrorOccurred()) {
    return;
  }

  const auto *ND = Result.Nodes.getNodeAs<NamedDecl>("decl");
  assert(ND && "Matched declaration must be a NamedDecl!");
  const SourceManager *SM = Result.SourceManager;

  llvm::Optional<SymbolInfo> Symbol =
      CreateSymbolInfo(ND, *SM, Collector);
  if (Symbol)
    Reporter->reportSymbol(
        SM->getFileEntryForID(SM->getMainFileID())->getName(), *Symbol);
}

} // namespace find_all_symbols
} // namespace clang
