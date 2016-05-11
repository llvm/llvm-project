//===-- FindAllSymbols.cpp - find all symbols -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FindAllSymbols.h"
#include "SymbolInfo.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::ast_matchers;

namespace clang {
namespace find_all_symbols {
namespace {
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
      Contexts.emplace_back(SymbolInfo::Namespace, NSD->isAnonymousNamespace()
                                                       ? ""
                                                       : NSD->getName().str());
    } else {
      const auto *RD = cast<RecordDecl>(Context);
      Contexts.emplace_back(SymbolInfo::Record, RD->getName().str());
    }
  }
  return Contexts;
}

llvm::Optional<SymbolInfo> CreateSymbolInfo(const NamedDecl *ND,
                                            const SourceManager &SM) {
  SymbolInfo::SymbolKind Type;
  if (llvm::isa<VarDecl>(ND)) {
    Type = SymbolInfo::Variable;
  } else if (llvm::isa<FunctionDecl>(ND)) {
    Type = SymbolInfo::Function;
  } else if (llvm::isa<TypedefNameDecl>(ND)) {
    Type = SymbolInfo::TypedefName;
  } else {
    assert(llvm::isa<RecordDecl>(ND) && "Matched decl must be one of VarDecl, "
                                        "FunctionDecl, TypedefNameDecl and "
                                        "RecordDecl!");
    // C-style record decl can have empty name, e.g "struct { ... } var;".
    if (ND->getName().empty())
      return llvm::None;
    Type = SymbolInfo::Class;
  }

  SourceLocation Loc = SM.getExpansionLoc(ND->getLocation());
  if (!Loc.isValid()) {
    llvm::errs() << "Declaration " << ND->getNameAsString() << "("
                 << ND->getDeclKindName()
                 << ") has invalid declaration location.";
    return llvm::None;
  }
  llvm::StringRef FilePath = SM.getFilename(Loc);
  if (FilePath.empty())
    return llvm::None;

  llvm::SmallString<128> AbsolutePath;
  if (llvm::sys::path::is_absolute(FilePath)) {
    AbsolutePath = FilePath;
  } else {
    auto WorkingDir = SM.getFileManager()
                          .getVirtualFileSystem()
                          ->getCurrentWorkingDirectory();
    if (!WorkingDir)
      return llvm::None;
    AbsolutePath = *WorkingDir;
    llvm::sys::path::append(AbsolutePath, FilePath);
  }

  llvm::sys::path::remove_dots(AbsolutePath, true);
  return SymbolInfo(ND->getNameAsString(), Type, AbsolutePath.str(),
                    GetContexts(ND), SM.getExpansionLineNumber(Loc));
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
            unless(isInstantiated()), unless(classTemplateSpecializationDecl()),
            unless(isExplicitTemplateSpecialization()));

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
      cxxRecordDecl(CommonFilter, CCMatcher, isDefinition(),
                    unless(isExplicitTemplateSpecialization()));
  MatchFinder->addMatcher(CxxRecordDecl.bind("decl"), this);

  // Matchers for function declarations.
  MatchFinder->addMatcher(
      functionDecl(CommonFilter, anyOf(ExternCMatcher, CCMatcher)).bind("decl"),
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
}

void FindAllSymbols::run(const MatchFinder::MatchResult &Result) {
  // Ignore Results in failing TUs.
  if (Result.Context->getDiagnostics().hasErrorOccurred()) {
    return;
  }

  const NamedDecl *ND = Result.Nodes.getNodeAs<NamedDecl>("decl");
  assert(ND && "Matched declaration must be a NamedDecl!");
  const SourceManager *SM = Result.SourceManager;

  llvm::Optional<SymbolInfo> Symbol = CreateSymbolInfo(ND, *SM);
  if (Symbol)
    Reporter->reportResult(
        SM->getFileEntryForID(SM->getMainFileID())->getName(), *Symbol);
}

} // namespace find_all_symbols
} // namespace clang
