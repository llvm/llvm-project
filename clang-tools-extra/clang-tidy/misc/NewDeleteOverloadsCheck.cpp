//===--- NewDeleteOverloadsCheck.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NewDeleteOverloadsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {

AST_MATCHER(FunctionDecl, isPlacementOverload) {
  bool New;
  switch (Node.getOverloadedOperator()) {
  default:
    return false;
  case OO_New:
  case OO_Array_New:
    New = true;
    break;
  case OO_Delete:
  case OO_Array_Delete:
    New = false;
    break;
  }

  // Variadic functions are always placement functions.
  if (Node.isVariadic())
    return true;

  // Placement new is easy: it always has more than one parameter (the first
  // parameter is always the size). If it's an overload of delete or delete[]
  // that has only one parameter, it's never a placement delete.
  if (New)
    return Node.getNumParams() > 1;
  if (Node.getNumParams() == 1)
    return false;

  // Placement delete is a little more challenging. They always have more than
  // one parameter with the first parameter being a pointer. However, the
  // second parameter can be a size_t for sized deallocation, and that is never
  // a placement delete operator.
  if (Node.getNumParams() <= 1 || Node.getNumParams() > 2)
    return true;

  const auto *FPT = Node.getType()->castAs<FunctionProtoType>();
  ASTContext &Ctx = Node.getASTContext();
  if (Ctx.getLangOpts().SizedDeallocation &&
      Ctx.hasSameType(FPT->getParamType(1), Ctx.getSizeType()))
    return false;

  return true;
}

OverloadedOperatorKind getCorrespondingOverload(const FunctionDecl *FD) {
  switch (FD->getOverloadedOperator()) {
  default:
    break;
  case OO_New:
    return OO_Delete;
  case OO_Delete:
    return OO_New;
  case OO_Array_New:
    return OO_Array_Delete;
  case OO_Array_Delete:
    return OO_Array_New;
  }
  llvm_unreachable("Not an overloaded allocation operator");
}

const char *getOperatorName(OverloadedOperatorKind K) {
  switch (K) {
  default:
    break;
  case OO_New:
    return "operator new";
  case OO_Delete:
    return "operator delete";
  case OO_Array_New:
    return "operator new[]";
  case OO_Array_Delete:
    return "operator delete[]";
  }
  llvm_unreachable("Not an overloaded allocation operator");
}

bool areCorrespondingOverloads(const FunctionDecl *LHS,
                               const FunctionDecl *RHS) {
  return RHS->getOverloadedOperator() == getCorrespondingOverload(LHS);
}

bool hasCorrespondingOverloadInBaseClass(const CXXMethodDecl *MD,
                                         const CXXRecordDecl *RD = nullptr) {
  if (RD) {
    // Check the methods in the given class and accessible to derived classes.
    for (const auto *BMD : RD->methods())
      if (BMD->isOverloadedOperator() && BMD->getAccess() != AS_private &&
          areCorrespondingOverloads(MD, BMD))
        return true;
  } else {
    // Get the parent class of the method; we do not need to care about checking
    // the methods in this class as the caller has already done that by looking
    // at the declaration contexts.
    RD = MD->getParent();
  }

  for (const auto &BS : RD->bases()) {
    // We can't say much about a dependent base class, but to avoid false
    // positives assume it can have a corresponding overload.
    if (BS.getType()->isDependentType())
      return true;
    if (const auto *BaseRD = BS.getType()->getAsCXXRecordDecl())
      if (hasCorrespondingOverloadInBaseClass(MD, BaseRD))
        return true;
  }

  return false;
}

} // anonymous namespace

void NewDeleteOverloadsCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  // Match all operator new and operator delete overloads (including the array
  // forms). Do not match implicit operators, placement operators, or
  // deleted/private operators.
  //
  // Technically, trivially-defined operator delete seems like a reasonable
  // thing to also skip. e.g., void operator delete(void *) {}
  // However, I think it's more reasonable to warn in this case as the user
  // should really be writing that as a deleted function.
  Finder->addMatcher(
      functionDecl(unless(anyOf(isImplicit(), isPlacementOverload(),
                                isDeleted(), cxxMethodDecl(isPrivate()))),
                   anyOf(hasOverloadedOperatorName("new"),
                         hasOverloadedOperatorName("new[]"),
                         hasOverloadedOperatorName("delete"),
                         hasOverloadedOperatorName("delete[]")))
          .bind("func"),
      this);
}

void NewDeleteOverloadsCheck::check(const MatchFinder::MatchResult &Result) {
  // Add any matches we locate to the list of things to be checked at the
  // end of the translation unit.
  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const CXXRecordDecl *RD = nullptr;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FD))
    RD = MD->getParent();
  Overloads[RD].push_back(FD);
}

void NewDeleteOverloadsCheck::onEndOfTranslationUnit() {
  // Walk over the list of declarations we've found to see if there is a
  // corresponding overload at the same declaration context or within a base
  // class. If there is not, add the element to the list of declarations to
  // diagnose.
  SmallVector<const FunctionDecl *, 4> Diagnose;
  for (const auto &RP : Overloads) {
    // We don't care about the CXXRecordDecl key in the map; we use it as a way
    // to shard the overloads by declaration context to reduce the algorithmic
    // complexity when searching for corresponding free store functions.
    for (const auto *Overload : RP.second) {
      const auto *Match =
          std::find_if(RP.second.begin(), RP.second.end(),
                       [&Overload](const FunctionDecl *FD) {
                         if (FD == Overload)
                           return false;
                         // If the declaration contexts don't match, we don't
                         // need to check any further.
                         if (FD->getDeclContext() != Overload->getDeclContext())
                           return false;

                         // Since the declaration contexts match, see whether
                         // the current element is the corresponding operator.
                         if (!areCorrespondingOverloads(Overload, FD))
                           return false;

                         return true;
                       });

      if (Match == RP.second.end()) {
        // Check to see if there is a corresponding overload in a base class
        // context. If there isn't, or if the overload is not a class member
        // function, then we should diagnose.
        const auto *MD = dyn_cast<CXXMethodDecl>(Overload);
        if (!MD || !hasCorrespondingOverloadInBaseClass(MD))
          Diagnose.push_back(Overload);
      }
    }
  }

  for (const auto *FD : Diagnose)
    diag(FD->getLocation(), "declaration of %0 has no matching declaration "
                            "of '%1' at the same scope")
        << FD << getOperatorName(getCorrespondingOverload(FD));
}

} // namespace misc
} // namespace tidy
} // namespace clang
