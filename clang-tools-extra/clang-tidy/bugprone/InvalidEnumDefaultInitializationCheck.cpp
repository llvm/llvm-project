//===--- InvalidEnumDefaultInitializationCheck.cpp - clang-tidy -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InvalidEnumDefaultInitializationCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <algorithm>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

bool isCompleteAndHasNoZeroValue(const EnumDecl *D) {
  const EnumDecl *Definition = D->getDefinition();
  return Definition && Definition->isComplete() &&
         !Definition->enumerators().empty() &&
         llvm::none_of(Definition->enumerators(),
                       [](const EnumConstantDecl *Value) {
                         return Value->getInitVal().isZero();
                       });
}

AST_MATCHER(EnumDecl, isCompleteAndHasNoZeroValue) {
  return isCompleteAndHasNoZeroValue(&Node);
}

// Find an initialization which initializes the value (if it has enum type) to a
// default zero value.
AST_MATCHER(Expr, isEmptyInit) {
  if (isa<CXXScalarValueInitExpr, ImplicitValueInitExpr>(&Node))
    return true;
  if (const auto *Init = dyn_cast<InitListExpr>(&Node)) {
    if (Init->getNumInits() == 0)
      return true;
  }
  return false;
}

AST_MATCHER(InitListExpr, hasArrayFiller) { return Node.hasArrayFiller(); }

// Check if any type has a "child" type that is an enum without zero value.
// The "child" type can be an array element type or member type of a record
// type (or a recursive combination of these). In this case, if the "root" type
// is statically initialized, the enum component is initialized to zero.
class FindEnumMember : public TypeVisitor<FindEnumMember, bool> {
public:
  const EnumType *FoundEnum = nullptr;

  bool VisitType(const Type *T) {
    const Type *DesT = T->getUnqualifiedDesugaredType();
    if (DesT != T)
      return Visit(DesT);
    return false;
  }
  bool VisitArrayType(const ArrayType *T) {
    return Visit(T->getElementType().getTypePtr());
  }
  bool VisitConstantArrayType(const ConstantArrayType *T) {
    return Visit(T->getElementType().getTypePtr());
  }
  bool VisitEnumType(const EnumType *T) {
    if (isCompleteAndHasNoZeroValue(T->getOriginalDecl())) {
      FoundEnum = T;
      return true;
    }
    return false;
  }
  bool VisitRecordType(const RecordType *T) {
    const RecordDecl *RD = T->getOriginalDecl()->getDefinition();
    if (!RD || RD->isUnion())
      return false;
    auto VisitField = [this](const FieldDecl *F) {
      return Visit(F->getType().getTypePtr());
    };
    return llvm::any_of(RD->fields(), VisitField);
  }
};

} // namespace

InvalidEnumDefaultInitializationCheck::InvalidEnumDefaultInitializationCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void InvalidEnumDefaultInitializationCheck::registerMatchers(
    MatchFinder *Finder) {
  auto EnumWithoutZeroValue = enumType(
      hasDeclaration(enumDecl(isCompleteAndHasNoZeroValue()).bind("enum")));
  auto EnumOrArrayOfEnum = qualType(hasUnqualifiedDesugaredType(
      anyOf(EnumWithoutZeroValue,
            arrayType(hasElementType(qualType(
                hasUnqualifiedDesugaredType(EnumWithoutZeroValue)))))));
  Finder->addMatcher(
      expr(isEmptyInit(), hasType(EnumOrArrayOfEnum)).bind("expr"), this);

  // Array initialization can contain an "array filler" for the (syntactically)
  // unspecified elements. This expression is not found by AST matchers and can
  // have any type (the array's element type). This is an implicitly generated
  // initialization, so if the type contains somewhere an enum without zero
  // enumerator, the zero initialization applies here. We search this array
  // element type for the specific enum type manually when this matcher matches.
  Finder->addMatcher(initListExpr(hasArrayFiller()).bind("array_filler_expr"),
                     this);
}

void InvalidEnumDefaultInitializationCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *InitExpr = Result.Nodes.getNodeAs<Expr>("expr");
  const auto *Enum = Result.Nodes.getNodeAs<EnumDecl>("enum");
  if (!InitExpr) {
    const auto *InitList =
        Result.Nodes.getNodeAs<InitListExpr>("array_filler_expr");
    // Initialization of omitted array elements with array filler was found.
    // Check the type for enum without zero value.
    // FIXME: In this way only one enum-typed value is found, not all of these.
    FindEnumMember Finder;
    if (!Finder.Visit(InitList->getArrayFiller()->getType().getTypePtr()))
      return;
    InitExpr = InitList;
    Enum = Finder.FoundEnum->getOriginalDecl();
  }

  if (!InitExpr || !Enum)
    return;

  ASTContext &ACtx = Enum->getASTContext();
  SourceLocation Loc = InitExpr->getExprLoc();
  if (Loc.isInvalid()) {
    if (isa<ImplicitValueInitExpr, InitListExpr>(InitExpr)) {
      DynTypedNodeList Parents = ACtx.getParents(*InitExpr);
      if (Parents.empty())
        return;

      if (const auto *Ctor = Parents[0].get<CXXConstructorDecl>()) {
        // Try to find member initializer with the found expression and get the
        // source location from it.
        CXXCtorInitializer *const *CtorInit = std::find_if(
            Ctor->init_begin(), Ctor->init_end(),
            [InitExpr](const CXXCtorInitializer *Init) {
              return Init->isMemberInitializer() && Init->getInit() == InitExpr;
            });
        if (!CtorInit)
          return;
        Loc = (*CtorInit)->getLParenLoc();
      } else if (const auto *InitList = Parents[0].get<InitListExpr>()) {
        // The expression may be implicitly generated for an initialization.
        // Search for a parent initialization list with valid source location.
        while (InitList->getExprLoc().isInvalid()) {
          DynTypedNodeList Parents = ACtx.getParents(*InitList);
          if (Parents.empty())
            return;
          InitList = Parents[0].get<InitListExpr>();
          if (!InitList)
            return;
        }
        Loc = InitList->getExprLoc();
      }
    }
    // If still not found a source location, omit the warning.
    // Ideally all such cases (if they exist) should be handled to make the
    // check more precise.
    if (Loc.isInvalid())
      return;
  }
  diag(Loc, "enum value of type %0 initialized with invalid value of 0, "
            "enum doesn't have a zero-value enumerator")
      << Enum;
  diag(Enum->getLocation(), "enum is defined here", DiagnosticIDs::Note);
}

} // namespace clang::tidy::bugprone
