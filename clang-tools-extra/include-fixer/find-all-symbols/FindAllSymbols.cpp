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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::ast_matchers;

namespace clang {
namespace find_all_symbols {
namespace {
void SetContext(const NamedDecl *ND, SymbolInfo *Symbol) {
  for (const auto *Context = ND->getDeclContext(); Context;
       Context = Context->getParent()) {
    if (llvm::isa<TranslationUnitDecl>(Context) ||
        llvm::isa<LinkageSpecDecl>(Context))
      break;

    assert(llvm::isa<NamedDecl>(Context) &&
           "Expect Context to be a NamedDecl");
    if (const auto *NSD = dyn_cast<NamespaceDecl>(Context)) {
      Symbol->Contexts.emplace_back(
              SymbolInfo::Namespace,
                  NSD->isAnonymousNamespace() ? "" : NSD->getName().str());
    } else {
      const auto *RD = cast<RecordDecl>(Context);
      Symbol->Contexts.emplace_back(SymbolInfo::Record, RD->getName().str());
    }
  }
}

bool SetCommonInfo(const MatchFinder::MatchResult &Result,
                   const NamedDecl *ND, SymbolInfo *Symbol) {
  SetContext(ND, Symbol);

  Symbol->Name = ND->getNameAsString();

  const SourceManager *SM = Result.SourceManager;
  SourceLocation Loc = SM->getExpansionLoc(ND->getLocation());
  if (!Loc.isValid()) {
    llvm::errs() << "Declaration " << ND->getNameAsString() << "("
                 << ND->getDeclKindName()
                 << ") has invalid declaration location.";
    return false;
  }

  Symbol->LineNumber = SM->getExpansionLineNumber(Loc);

  llvm::StringRef FilePath = SM->getFilename(Loc);
  if (FilePath.empty())
    return false;

  llvm::SmallString<128> AbsolutePath;
  if (llvm::sys::path::is_absolute(FilePath)) {
    AbsolutePath = FilePath;
  } else {
    auto WorkingDir = SM->getFileManager()
                          .getVirtualFileSystem()
                          ->getCurrentWorkingDirectory();
    if (!WorkingDir)
      return false;
    AbsolutePath = *WorkingDir;
    llvm::sys::path::append(AbsolutePath, FilePath);
  }

  llvm::sys::path::remove_dots(AbsolutePath, true);
  Symbol->FilePath = AbsolutePath.str();
  return true;
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

  SymbolInfo Symbol;
  if (!SetCommonInfo(Result, ND, &Symbol))
    return;

  if (const auto *VD = llvm::dyn_cast<VarDecl>(ND)) {
    Symbol.Type = SymbolInfo::Variable;
    SymbolInfo::VariableInfo VI;
    VI.Type = VD->getType().getAsString();
    Symbol.VariableInfos = VI;
  } else if (const auto *FD = llvm::dyn_cast<FunctionDecl>(ND)) {
    Symbol.Type = SymbolInfo::Function;
    SymbolInfo::FunctionInfo FI;
    FI.ReturnType = FD->getReturnType().getAsString();
    for (const auto *Param : FD->params())
      FI.ParameterTypes.push_back(Param->getType().getAsString());
    Symbol.FunctionInfos = FI;
  } else if (const auto *TD = llvm::dyn_cast<TypedefNameDecl>(ND)) {
    Symbol.Type = SymbolInfo::TypedefName;
    SymbolInfo::TypedefNameInfo TI;
    TI.UnderlyingType = TD->getUnderlyingType().getAsString();
    Symbol.TypedefNameInfos = TI;
  } else {
    assert(
        llvm::isa<RecordDecl>(ND) &&
        "Matched decl must be one of VarDecl, FunctionDecl, and RecordDecl!");
    // C-style record decl can have empty name, e.g "struct { ... } var;".
    if (ND->getName().empty())
      return;
    Symbol.Type = SymbolInfo::Class;
  }

  const FileEntry *FE = SM->getFileEntryForID(SM->getMainFileID());
  Reporter->reportResult(FE->getName(), Symbol);
}

} // namespace find_all_symbols
} // namespace clang
