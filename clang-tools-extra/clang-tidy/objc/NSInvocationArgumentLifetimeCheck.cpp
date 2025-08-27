//===--- NSInvocationArgumentLifetimeCheck.cpp - clang-tidy ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NSInvocationArgumentLifetimeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ComputeDependence.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/TypeBase.h"
#include "clang/AST/TypeLoc.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::objc {

static constexpr StringRef WeakText = "__weak";
static constexpr StringRef StrongText = "__strong";
static constexpr StringRef UnsafeUnretainedText = "__unsafe_unretained";

namespace {

/// Matches ObjCIvarRefExpr, DeclRefExpr, or MemberExpr that reference
/// Objective-C object (or block) variables or fields whose object lifetimes
/// are not __unsafe_unretained.
AST_POLYMORPHIC_MATCHER(isObjCManagedLifetime,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(ObjCIvarRefExpr,
                                                        DeclRefExpr,
                                                        MemberExpr)) {
  QualType QT = Node.getType();
  return QT->isScalarType() &&
         (QT->getScalarTypeKind() == Type::STK_ObjCObjectPointer ||
          QT->getScalarTypeKind() == Type::STK_BlockPointer) &&
         QT.getQualifiers().getObjCLifetime() > Qualifiers::OCL_ExplicitNone;
}

} // namespace

static std::optional<FixItHint>
fixItHintReplacementForOwnershipString(StringRef Text, CharSourceRange Range,
                                       StringRef Ownership) {
  size_t Index = Text.find(Ownership);
  if (Index == StringRef::npos)
    return std::nullopt;

  SourceLocation Begin = Range.getBegin().getLocWithOffset(Index);
  SourceLocation End = Begin.getLocWithOffset(Ownership.size());
  return FixItHint::CreateReplacement(SourceRange(Begin, End),
                                      UnsafeUnretainedText);
}

static std::optional<FixItHint>
fixItHintForVarDecl(const VarDecl *VD, const SourceManager &SM,
                    const LangOptions &LangOpts) {
  assert(VD && "VarDecl parameter must not be null");
  // Don't provide fix-its for any parameter variables at this time.
  if (isa<ParmVarDecl>(VD))
    return std::nullopt;

  // Currently there is no way to directly get the source range for the
  // __weak/__strong ObjC lifetime qualifiers, so it's necessary to string
  // search in the source code.
  CharSourceRange Range = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(VD->getSourceRange()), SM, LangOpts);
  if (Range.isInvalid()) {
    // An invalid range likely means inside a macro, in which case don't supply
    // a fix-it.
    return std::nullopt;
  }

  StringRef VarDeclText = Lexer::getSourceText(Range, SM, LangOpts);
  if (std::optional<FixItHint> Hint =
          fixItHintReplacementForOwnershipString(VarDeclText, Range, WeakText))
    return Hint;

  if (std::optional<FixItHint> Hint = fixItHintReplacementForOwnershipString(
          VarDeclText, Range, StrongText))
    return Hint;

  return FixItHint::CreateInsertion(Range.getBegin(), "__unsafe_unretained ");
}

void NSInvocationArgumentLifetimeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          objcMessageExpr(
              hasReceiverType(asString("NSInvocation *")),
              anyOf(hasSelector("getArgument:atIndex:"),
                    hasSelector("getReturnValue:")),
              hasArgument(
                  0,
                  anyOf(hasDescendant(memberExpr(isObjCManagedLifetime())),
                        hasDescendant(objcIvarRefExpr(isObjCManagedLifetime())),
                        hasDescendant(
                            // Reference to variables, but when dereferencing
                            // to ivars/fields a more-descendent variable
                            // reference (e.g. self) may match with strong
                            // object lifetime, leading to an incorrect match.
                            // Exclude these conditions.
                            declRefExpr(to(varDecl().bind("var")),
                                        unless(hasParent(implicitCastExpr())),
                                        isObjCManagedLifetime())))))
              .bind("call")),
      this);
}

void NSInvocationArgumentLifetimeCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedExpr = Result.Nodes.getNodeAs<ObjCMessageExpr>("call");

  auto Diag = diag(MatchedExpr->getArg(0)->getBeginLoc(),
                   "NSInvocation %objcinstance0 should only pass pointers to "
                   "objects with ownership __unsafe_unretained")
              << MatchedExpr->getSelector();

  // Only provide fix-it hints for references to local variables; fixes for
  // instance variable references don't have as clear an automated fix.
  const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var");
  if (!VD)
    return;

  if (auto Hint = fixItHintForVarDecl(VD, *Result.SourceManager,
                                      Result.Context->getLangOpts()))
    Diag << *Hint;
}

} // namespace clang::tidy::objc
