//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SmartPtrInitializationCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

const auto DefaultSharedPointers = "::std::shared_ptr;::boost::shared_ptr";
const auto DefaultUniquePointers = "::std::unique_ptr";
const auto DefaultDefaultDeleters = "::std::default_delete";

} // namespace

SmartPtrInitializationCheck::SmartPtrInitializationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SharedPointers(utils::options::parseStringList(
          Options.get("SharedPointers", DefaultSharedPointers))),
      UniquePointers(utils::options::parseStringList(
          Options.get("UniquePointers", DefaultUniquePointers))),
      DefaultDeleters(utils::options::parseStringList(
          Options.get("DefaultDeleters", DefaultDefaultDeleters))) {}

void SmartPtrInitializationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SharedPointers",
                utils::options::serializeStringList(SharedPointers));
  Options.store(Opts, "UniquePointers",
                utils::options::serializeStringList(UniquePointers));
  Options.store(Opts, "DefaultDeleters",
                utils::options::serializeStringList(DefaultDeleters));
}

void SmartPtrInitializationCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsSharedPtr = hasAnyName(SharedPointers);
  const auto IsUniquePtr = hasAnyName(UniquePointers);
  const auto IsSmartPtr = anyOf(IsSharedPtr, IsUniquePtr);
  const auto IsDefaultDeleter = hasAnyName(DefaultDeleters);

  const auto IsSharedPtrRecord = cxxRecordDecl(IsSharedPtr);
  const auto IsUniquePtrRecord = cxxRecordDecl(IsUniquePtr);
  const auto IsSmartPtrRecord = cxxRecordDecl(IsSmartPtr);

  auto ReleaseCallMatcher =
      cxxMemberCallExpr(callee(cxxMethodDecl(hasName("release"))));

  // Array automatically decays to pointer
  auto PointerArg = expr(anyOf(hasType(pointerType()), hasType(arrayType())))
                        .bind("pointer-arg");

  // Matcher for unique_ptr types with custom deleters
  auto UniquePtrWithCustomDeleter = classTemplateSpecializationDecl(
      IsUniquePtr, templateArgumentCountIs(2),
      hasTemplateArgument(
          1, refersToType(
                 unless(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                     classTemplateSpecializationDecl(IsDefaultDeleter))))))));

  // Matcher for smart pointer constructors
  // Exclude constructors with custom deleters:
  // - shared_ptr with 2+ arguments (second is deleter)
  // - unique_ptr with 2+ template args where second is not default_delete
  auto HasCustomDeleter = anyOf(
      allOf(hasDeclaration(cxxConstructorDecl(ofClass(IsSharedPtrRecord))),
            hasArgument(1, anything())),
      allOf(hasType(hasUnqualifiedDesugaredType(
                recordType(hasDeclaration(UniquePtrWithCustomDeleter)))),
            hasDeclaration(cxxConstructorDecl(ofClass(IsUniquePtrRecord)))));

  // FIXME: need proper suppurt for conditionalOperator cases
  auto AllowedArguments = anyOf(ignoringParenCasts(cxxNewExpr()),
                                ignoringParenCasts(ReleaseCallMatcher),
                                ignoringParenCasts(conditionalOperator()));

  auto RawPtrMatcher =
      declRefExpr(to(varDecl(hasInitializer(ignoringParenCasts(cxxNewExpr())))
                         .bind("raw-ptr")));

  auto SmartPtrConstructorMatcher = cxxConstructExpr(
      hasDeclaration(
          cxxConstructorDecl(ofClass(IsSmartPtrRecord.bind("method-parent")))
              .bind("method-decl")),
      hasArgument(0, PointerArg), unless(HasCustomDeleter),
      unless(hasArgument(0, AllowedArguments)),
      optionally(hasArgument(0, RawPtrMatcher)));

  // Matcher for reset() calls
  // Exclude reset() calls with custom deleters:
  // - shared_ptr with 2+ arguments (second is deleter)
  // - unique_ptr with custom deleter type (2+ template args where second is not
  // default_delete)
  auto HasCustomDeleterInReset = anyOf(
      allOf(on(hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                classTemplateSpecializationDecl(IsSharedPtr)))))),
            hasArgument(1, anything())),
      on(hasType(hasUnqualifiedDesugaredType(
          recordType(hasDeclaration(UniquePtrWithCustomDeleter))))));

  auto ResetCallMatcher = cxxMemberCallExpr(
      on(hasType(hasUnqualifiedDesugaredType(recordType(
          hasDeclaration(classTemplateSpecializationDecl(IsSmartPtr)))))),
      callee(cxxMethodDecl(ofClass(IsSmartPtrRecord.bind("method-parent")),
                           hasName("reset"))
                 .bind("method-decl")),
      hasArgument(0, PointerArg), unless(HasCustomDeleterInReset),
      unless(hasArgument(0, AllowedArguments)),
      optionally(hasArgument(0, RawPtrMatcher)));

  Finder->addMatcher(SmartPtrConstructorMatcher, this);
  Finder->addMatcher(ResetCallMatcher, this);
}

void SmartPtrInitializationCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *PointerArg = Result.Nodes.getNodeAs<Expr>("pointer-arg");
  const auto *MethodDecl = Result.Nodes.getNodeAs<CXXMethodDecl>("method-decl");
  const auto *Record = Result.Nodes.getNodeAs<CXXRecordDecl>("method-parent");
  const auto *RawPtrVar = Result.Nodes.getNodeAs<VarDecl>("raw-ptr");

  if (!MethodDecl)
    return;

  assert(PointerArg && Record);

  if (RawPtrVar) {
    // Store information about the raw pointer and its initializations
    // Use the raw pointer as a key
    const VarDecl *Key = RawPtrVar;
    const unsigned InitsCount = ++SmartPtrInitMap[Key];

    // Check if this raw pointer was used to initialize
    // multiple smart pointers in one function
    if (InitsCount <= 1)
      return;
  }

  const SourceLocation Loc = PointerArg->getBeginLoc();
  if (Loc.isInvalid())
    return;

  diag(Loc, "passing a raw pointer '%0' to '%1%2 may cause double deletion")
      << getRawPointerDescription(PointerArg, *Result.Context)
      << getSmartPointerDescription(Record, *Result.Context)
      << (isa<CXXConstructorDecl>(MethodDecl) ? "' constructor" : "::reset'");
}

std::string SmartPtrInitializationCheck::getSmartPointerDescription(
    const CXXRecordDecl *RecordDecl, const ASTContext &Context) {
  const PrintingPolicy Policy = Context.getPrintingPolicy();

  std::string Result;
  llvm::raw_string_ostream OS(Result);
  RecordDecl->getNameForDiagnostic(OS, Policy, /*Qualified=*/true);

  return Result;
}

std::string SmartPtrInitializationCheck::getRawPointerDescription(
    const Expr *PointerExpr, const ASTContext &Context) {
  const QualType ExprType = PointerExpr->getType();

  PrintingPolicy Policy(Context.getLangOpts());
  Policy.SuppressSpecifiers = false;
  Policy.SuppressTagKeyword = true;

  std::string Result = ExprType.getAsString(Policy);

  size_t Pos = Result.find(" *");
  while (Pos != std::string::npos) {
    Result.erase(Pos, 1); // remove the space
    Pos = Result.find(" *", Pos);
  }

  return Result;
}

} // namespace clang::tidy::bugprone
