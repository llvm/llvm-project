//===--- QualifiedAutoCheck.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QualifiedAutoCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {

// FIXME move to ASTMatchers
AST_MATCHER_P(QualType, hasUnqualifiedType,
              ast_matchers::internal::Matcher<QualType>, InnerMatcher) {
  return InnerMatcher.matches(Node.getUnqualifiedType(), Finder, Builder);
}

enum class Qualifier { Const, Volatile, Restrict };

llvm::Optional<Token> findQualToken(const VarDecl *Decl, Qualifier Qual,
                                    const MatchFinder::MatchResult &Result) {
  // Since either of the locs can be in a macro, use `makeFileCharRange` to be
  // sure that we have a consistent `CharSourceRange`, located entirely in the
  // source file.

  assert((Qual == Qualifier::Const || Qual == Qualifier::Volatile ||
          Qual == Qualifier::Restrict) &&
         "Invalid Qualifier");

  SourceLocation BeginLoc = Decl->getQualifierLoc().getBeginLoc();
  if (BeginLoc.isInvalid())
    BeginLoc = Decl->getBeginLoc();
  SourceLocation EndLoc = Decl->getLocation();

  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(BeginLoc, EndLoc), *Result.SourceManager,
      Result.Context->getLangOpts());

  if (FileRange.isInvalid())
    return llvm::None;

  tok::TokenKind Tok =
      Qual == Qualifier::Const
          ? tok::kw_const
          : Qual == Qualifier::Volatile ? tok::kw_volatile : tok::kw_restrict;

  return utils::lexer::getQualifyingToken(Tok, FileRange, *Result.Context,
                                          *Result.SourceManager);
}

llvm::Optional<SourceRange>
getTypeSpecifierLocation(const VarDecl *Var,
                         const MatchFinder::MatchResult &Result) {
  SourceRange TypeSpecifier(
      Var->getTypeSpecStartLoc(),
      Var->getTypeSpecEndLoc().getLocWithOffset(Lexer::MeasureTokenLength(
          Var->getTypeSpecEndLoc(), *Result.SourceManager,
          Result.Context->getLangOpts())));

  if (TypeSpecifier.getBegin().isMacroID() ||
      TypeSpecifier.getEnd().isMacroID())
    return llvm::None;
  return TypeSpecifier;
}

llvm::Optional<SourceRange> mergeReplacementRange(SourceRange &TypeSpecifier,
                                                  const Token &ConstToken) {
  if (TypeSpecifier.getBegin().getLocWithOffset(-1) == ConstToken.getEndLoc()) {
    TypeSpecifier.setBegin(ConstToken.getLocation());
    return llvm::None;
  }
  if (TypeSpecifier.getEnd().getLocWithOffset(1) == ConstToken.getLocation()) {
    TypeSpecifier.setEnd(ConstToken.getEndLoc());
    return llvm::None;
  }
  return SourceRange(ConstToken.getLocation(), ConstToken.getEndLoc());
}

bool isPointerConst(QualType QType) {
  QualType Pointee = QType->getPointeeType();
  assert(!Pointee.isNull() && "can't have a null Pointee");
  return Pointee.isConstQualified();
}

bool isAutoPointerConst(QualType QType) {
  QualType Pointee =
      cast<AutoType>(QType->getPointeeType().getTypePtr())->desugar();
  assert(!Pointee.isNull() && "can't have a null Pointee");
  return Pointee.isConstQualified();
}

} // namespace

void QualifiedAutoCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AddConstToQualified", AddConstToQualified);
}

void QualifiedAutoCheck::registerMatchers(MatchFinder *Finder) {
  auto ExplicitSingleVarDecl =
      [](const ast_matchers::internal::Matcher<VarDecl> &InnerMatcher,
         llvm::StringRef ID) {
        return declStmt(
            unless(isInTemplateInstantiation()),
            hasSingleDecl(
                varDecl(unless(isImplicit()), InnerMatcher).bind(ID)));
      };
  auto ExplicitSingleVarDeclInTemplate =
      [](const ast_matchers::internal::Matcher<VarDecl> &InnerMatcher,
         llvm::StringRef ID) {
        return declStmt(
            isInTemplateInstantiation(),
            hasSingleDecl(
                varDecl(unless(isImplicit()), InnerMatcher).bind(ID)));
      };

  auto IsBoundToType = refersToType(equalsBoundNode("type"));

  Finder->addMatcher(
      ExplicitSingleVarDecl(hasType(autoType(hasDeducedType(
                                pointerType(pointee(unless(functionType())))))),
                            "auto"),
      this);

  Finder->addMatcher(
      ExplicitSingleVarDeclInTemplate(
          allOf(hasType(autoType(hasDeducedType(pointerType(
                    pointee(hasUnqualifiedType(qualType().bind("type")),
                            unless(functionType())))))),
                anyOf(hasAncestor(
                          functionDecl(hasAnyTemplateArgument(IsBoundToType))),
                      hasAncestor(classTemplateSpecializationDecl(
                          hasAnyTemplateArgument(IsBoundToType))))),
          "auto"),
      this);
  if (!AddConstToQualified)
    return;
  Finder->addMatcher(ExplicitSingleVarDecl(
                         hasType(pointerType(pointee(autoType()))), "auto_ptr"),
                     this);
  Finder->addMatcher(
      ExplicitSingleVarDecl(hasType(lValueReferenceType(pointee(autoType()))),
                            "auto_ref"),
      this);
}

void QualifiedAutoCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("auto")) {
    SourceRange TypeSpecifier;
    if (llvm::Optional<SourceRange> TypeSpec =
            getTypeSpecifierLocation(Var, Result)) {
      TypeSpecifier = *TypeSpec;
    } else
      return;

    llvm::SmallVector<SourceRange, 4> RemoveQualifiersRange;
    auto CheckQualifier = [&](bool IsPresent, Qualifier Qual) {
      if (IsPresent) {
        llvm::Optional<Token> Token = findQualToken(Var, Qual, Result);
        if (!Token || Token->getLocation().isMacroID())
          return true; // Disregard this VarDecl.
        if (llvm::Optional<SourceRange> Result =
                mergeReplacementRange(TypeSpecifier, *Token))
          RemoveQualifiersRange.push_back(*Result);
      }
      return false;
    };

    bool IsLocalConst = Var->getType().isLocalConstQualified();
    bool IsLocalVolatile = Var->getType().isLocalVolatileQualified();
    bool IsLocalRestrict = Var->getType().isLocalRestrictQualified();

    if (CheckQualifier(IsLocalConst, Qualifier::Const) ||
        CheckQualifier(IsLocalVolatile, Qualifier::Volatile) ||
        CheckQualifier(IsLocalRestrict, Qualifier::Restrict))
      return;

    // Check for bridging the gap between the asterisk and name.
    if (Var->getLocation() == TypeSpecifier.getEnd().getLocWithOffset(1))
      TypeSpecifier.setEnd(TypeSpecifier.getEnd().getLocWithOffset(1));

    CharSourceRange FixItRange = CharSourceRange::getCharRange(TypeSpecifier);
    if (FixItRange.isInvalid())
      return;

    SourceLocation FixitLoc = FixItRange.getBegin();
    for (SourceRange &Range : RemoveQualifiersRange) {
      if (Range.getBegin() < FixitLoc)
        FixitLoc = Range.getBegin();
    }

    std::string ReplStr = [&] {
      llvm::StringRef PtrConst = isPointerConst(Var->getType()) ? "const " : "";
      llvm::StringRef LocalConst = IsLocalConst ? "const " : "";
      llvm::StringRef LocalVol = IsLocalVolatile ? "volatile " : "";
      llvm::StringRef LocalRestrict = IsLocalRestrict ? "__restrict " : "";
      return (PtrConst + "auto *" + LocalConst + LocalVol + LocalRestrict)
          .str();
    }();

    DiagnosticBuilder Diag =
        diag(FixitLoc, "'%0%1%2auto %3' can be declared as '%4%3'")
        << (IsLocalConst ? "const " : "")
        << (IsLocalVolatile ? "volatile " : "")
        << (IsLocalRestrict ? "__restrict " : "") << Var->getName() << ReplStr;

    for (SourceRange &Range : RemoveQualifiersRange) {
      Diag << FixItHint::CreateRemoval(CharSourceRange::getCharRange(Range));
    }

    Diag << FixItHint::CreateReplacement(FixItRange, ReplStr);
    return;
  }
  if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("auto_ptr")) {
    if (!isPointerConst(Var->getType()))
      return; // Pointer isn't const, no need to add const qualifier.
    if (!isAutoPointerConst(Var->getType()))
      return; // Const isnt wrapped in the auto type, so must be declared
              // explicitly.

    if (Var->getType().isLocalConstQualified()) {
      llvm::Optional<Token> Token =
          findQualToken(Var, Qualifier::Const, Result);
      if (!Token || Token->getLocation().isMacroID())
        return;
    }
    if (Var->getType().isLocalVolatileQualified()) {
      llvm::Optional<Token> Token =
          findQualToken(Var, Qualifier::Volatile, Result);
      if (!Token || Token->getLocation().isMacroID())
        return;
    }
    if (Var->getType().isLocalRestrictQualified()) {
      llvm::Optional<Token> Token =
          findQualToken(Var, Qualifier::Restrict, Result);
      if (!Token || Token->getLocation().isMacroID())
        return;
    }

    if (llvm::Optional<SourceRange> TypeSpec =
            getTypeSpecifierLocation(Var, Result)) {
      if (TypeSpec->isInvalid() || TypeSpec->getBegin().isMacroID() ||
          TypeSpec->getEnd().isMacroID())
        return;
      SourceLocation InsertPos = TypeSpec->getBegin();
      diag(InsertPos, "'auto *%0%1%2' can be declared as 'const auto *%0%1%2'")
          << (Var->getType().isLocalConstQualified() ? "const " : "")
          << (Var->getType().isLocalVolatileQualified() ? "volatile " : "")
          << Var->getName() << FixItHint::CreateInsertion(InsertPos, "const ");
    }
    return;
  }
  if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("auto_ref")) {
    if (!isPointerConst(Var->getType()))
      return; // Pointer isn't const, no need to add const qualifier.
    if (!isAutoPointerConst(Var->getType()))
      // Const isnt wrapped in the auto type, so must be declared explicitly.
      return;

    if (llvm::Optional<SourceRange> TypeSpec =
            getTypeSpecifierLocation(Var, Result)) {
      if (TypeSpec->isInvalid() || TypeSpec->getBegin().isMacroID() ||
          TypeSpec->getEnd().isMacroID())
        return;
      SourceLocation InsertPos = TypeSpec->getBegin();
      diag(InsertPos, "'auto &%0' can be declared as 'const auto &%0'")
          << Var->getName() << FixItHint::CreateInsertion(InsertPos, "const ");
    }
    return;
  }
}

} // namespace readability
} // namespace tidy
} // namespace clang
