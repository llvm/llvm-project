//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseBracedInitializationCheck.h"
#include "../utils/LexerUtils.h"
#include "../utils/NarrowingConversions.h"
#include "clang/AST/ASTContext.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

/// Returns \c true if \p From may be implicitly converted to \p To.
static bool mayConvertImplicitly(QualType From, QualType To) {
  From = From.getNonReferenceType().getCanonicalType();
  To = To.getNonReferenceType().getCanonicalType();
  if (From == To)
    return true;

  if ((From->isPointerType() || From->isArrayType()) &&
      To->isArithmeticType() && !To->isBooleanType())
    return false;

  if (const auto *FromEnum = From->getAs<EnumType>())
    if (FromEnum->getDecl()->isScoped())
      return false;

  if (const auto *FromRecord = From->getAsCXXRecordDecl();
      FromRecord && !To->isRecordType()) {
    if (!FromRecord->hasDefinition())
      return true;
    return !FromRecord->getVisibleConversionFunctions().empty();
  }

  return true;
}

/// Returns the prefix of \p Ctor 's arguments that are explicitly written.
/// Default arguments always sit at the tail of the argument list.
static ArrayRef<const Expr *> getExplicitArgs(const CXXConstructExpr &Ctor) {
  ArrayRef<const Expr *> Args(Ctor.getArgs(), Ctor.getNumArgs());
  while (!Args.empty() && isa<CXXDefaultArgExpr>(Args.back()))
    Args = Args.drop_back();
  return Args;
}

static bool hasInitListCtor(const CXXRecordDecl *RD,
                            ArrayRef<const Expr *> ExplicitArgs) {
  if (!RD || !RD->hasDefinition())
    return false;

  for (const CXXConstructorDecl *CD : RD->ctors()) {
    if (CD->getNumParams() == 0)
      continue;
    const QualType FirstParam =
        CD->getParamDecl(0)->getType().getNonReferenceType();
    const auto *Init = FirstParam->getAsCXXRecordDecl();
    if (!Init || !Init->getDeclName().isIdentifier() ||
        Init->getName() != "initializer_list" || !Init->isInStdNamespace())
      continue;
    // [dcl.init.list] p2: all other params must have defaults.
    bool OthersDefaulted = true;
    for (unsigned I = 1; I < CD->getNumParams(); ++I)
      if (!CD->getParamDecl(I)->hasDefaultArg()) {
        OthersDefaulted = false;
        break;
      }
    if (!OthersDefaulted)
      continue;
    const auto *InitSpec = dyn_cast<ClassTemplateSpecializationDecl>(Init);
    if (!InitSpec || InitSpec->getTemplateArgs().size() < 1)
      return true;
    const QualType InitType = InitSpec->getTemplateArgs()[0].getAsType();

    if (llvm::all_of(ExplicitArgs, [&InitType](const Expr *Arg) {
          return mayConvertImplicitly(Arg->getType(), InitType);
        }))
      return true;
  }

  return false;
}

namespace {

AST_MATCHER_P(VarDecl, hasInitStyle, VarDecl::InitializationStyle, Style) {
  return Node.getInitStyle() == Style;
}

AST_MATCHER(Type, isDependentType) { return Node.isDependentType(); }

AST_MATCHER(CXXConstructExpr, noMacroParens) {
  const SourceRange Range = Node.getParenOrBraceRange();
  return Range.isValid() && !Range.getBegin().isMacroID() &&
         !Range.getEnd().isMacroID();
}

const ast_matchers::internal::VariadicDynCastAllOfMatcher<Stmt,
                                                          CXXParenListInitExpr>
    CxxParenListInitExpr;

/// Matches 'CXXConstructExpr' whose target class has any constructor taking
/// 'std::initializer_list<Type>' where all arguments of the current call could
/// be converted to 'Type'.
AST_MATCHER(CXXConstructExpr, canOverlapWithInitListCtor) {
  const CXXRecordDecl *RD = Node.getConstructor()->getParent();
  assert(RD && "CXXConstructExpr must have a parent CXXRecordDecl");
  return hasInitListCtor(RD, getExplicitArgs(Node));
}

struct ParenRange {
  SourceLocation DiagLoc;
  SourceLocation LParen;
  SourceLocation RParen;
};

struct NarrowingInfo {
  SourceLocation Loc;
  QualType From;
  QualType To;
};

} // namespace

static std::optional<NarrowingInfo>
checkNarrowing(const Expr *Init, QualType TargetType, const ASTContext &Ctx) {
  const Expr *OrigInit = Init->IgnoreImpCasts();
  const QualType From = OrigInit->getType();
  if (From.isNull())
    return std::nullopt;
  const QualType To = TargetType.getNonReferenceType();
  if (utils::isNarrowingConversion(From, To, OrigInit, Ctx))
    return NarrowingInfo{OrigInit->getBeginLoc(), From, To};
  return std::nullopt;
}

/// Check whether a scalar initialization expression is narrowing.
static std::optional<NarrowingInfo> isScalarNarrowing(const Expr *Init,
                                                      QualType TargetType,
                                                      const ASTContext &Ctx) {
  return checkNarrowing(Init, TargetType, Ctx);
}

/// Returns a NarrowingInfo for every argument of \p Ctor that would narrow
/// under braced initialization. Empty if no argument narrows.
static SmallVector<NarrowingInfo> isCtorNarrowing(const CXXConstructExpr *Ctor,
                                                  const ASTContext &Ctx) {
  SmallVector<NarrowingInfo> Result;
  const CXXConstructorDecl *CD = Ctor->getConstructor();
  for (unsigned I = 0; I < Ctor->getNumArgs(); ++I) {
    const Expr *Arg = Ctor->getArg(I);
    if (isa<CXXDefaultArgExpr>(Arg))
      continue;
    if (I >= CD->getNumParams())
      break;
    if (auto Info = checkNarrowing(Arg, CD->getParamDecl(I)->getType(), Ctx))
      Result.push_back(*Info);
  }
  return Result;
}

/// Returns a NarrowingInfo for every user-specified initializer in \p PLE
/// that would narrow. Empty if none narrow.
static SmallVector<NarrowingInfo>
isPLENarrowing(const CXXParenListInitExpr *PLE, const ASTContext &Ctx) {
  SmallVector<NarrowingInfo> Result;
  for (const Expr *Init : PLE->getUserSpecifiedInitExprs())
    if (auto Info = checkNarrowing(Init, Init->getType(), Ctx))
      Result.push_back(*Info);
  return Result;
}

/// Locates the parentheses of a scalar or decomposition variable declaration
/// initialized with call syntax, e.g. 'int x(42)' or 'auto [a, b](expr)'.
static std::optional<ParenRange> handleScalarVar(const VarDecl *Var,
                                                 const SourceManager &SM,
                                                 const LangOptions &LangOpts) {
  const Expr *Init = Var->getInit();
  SourceLocation InitBegin = Init->getBeginLoc();
  SourceLocation InitEnd = Init->getEndLoc();
  if (isa<DecompositionDecl>(Var))
    if (const auto *Ctor = dyn_cast<CXXConstructExpr>(Init);
        Ctor && Ctor->getNumArgs() == 1 && InitBegin == Var->getLocation()) {
      const Expr *Arg = Ctor->getArg(0);
      InitBegin = Arg->getBeginLoc();
      InitEnd = Arg->getEndLoc();
    }
  const std::optional<Token> LTok =
      utils::lexer::findPreviousTokenSkippingComments(InitBegin, SM, LangOpts);
  if (!LTok || LTok->isNot(tok::l_paren) || LTok->getLocation().isMacroID())
    return std::nullopt;
  const std::optional<Token> RTok =
      utils::lexer::findNextTokenSkippingComments(InitEnd, SM, LangOpts);
  if (!RTok || RTok->isNot(tok::r_paren) || RTok->getLocation().isMacroID())
    return std::nullopt;
  return ParenRange{Var->getLocation(), LTok->getLocation(),
                    RTok->getLocation()};
}

/// Computes the parenthesis range for a parenthesized list initialization,
/// dispatching on which context (variable declaration, ...) it appears in.
static std::optional<ParenRange>
handlePLE(const CXXParenListInitExpr *PLE,
          const MatchFinder::MatchResult &Result) {
  SourceLocation DiagLoc;
  if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var_ple"))
    DiagLoc = Var->getLocation();
  else
    return std::nullopt;
  return ParenRange{DiagLoc, PLE->getBeginLoc(), PLE->getEndLoc()};
}

void UseBracedInitializationCheck::registerMatchers(MatchFinder *Finder) {
  // The C++ Core Guidelines rule ES.23 only targets variable declarations:
  // "Flag uses of () initialization syntax that are actually declarations."
  const auto GoodCtor =
      allOf(noMacroParens(), unless(canOverlapWithInitListCtor()),
            unless(isListInitialization()));
  const auto GoodCtorExpr = cxxConstructExpr(GoodCtor).bind("ctor");
  const auto GoodVar =
      allOf(unless(hasType(isDependentType())), unless(hasType(autoType())));
  const auto HasGoodCtorOrIsScalar =
      anyOf(hasInitializer(ignoringImplicit(GoodCtorExpr)),
            unless(hasInitializer(ignoringImplicit(cxxConstructExpr()))));

  Finder->addMatcher(varDecl(unless(decompositionDecl()),
                             hasInitStyle(VarDecl::CallInit), GoodVar,
                             HasGoodCtorOrIsScalar)
                         .bind("var"),
                     this);
  Finder->addMatcher(decompositionDecl(hasInitStyle(VarDecl::CallInit),
                                       unless(hasType(isDependentType())))
                         .bind("var"),
                     this);

  // C++20 parenthesized aggregate initialization of a variable, e.g.
  // 'Aggregate a(1, 2)' or 'int arr[3](1, 2, 3)'.
  if (getLangOpts().CPlusPlus20) {
    Finder->addMatcher(
        CxxParenListInitExpr(
            hasParent(varDecl(hasInitStyle(VarDecl::ParenListInit), GoodVar)
                          .bind("var_ple")))
            .bind("ple"),
        this);
  }
}

namespace {
struct MatchAnalysis {
  std::optional<ParenRange> Range;
  SmallVector<NarrowingInfo> Narrowings;
};
} // namespace

static MatchAnalysis analyzeMatch(const MatchFinder::MatchResult &Result,
                                  const ASTContext &Ctx,
                                  const SourceManager &SM,
                                  const LangOptions &LangOpts) {
  const auto ScalarNarrowing =
      [&Ctx](const Expr *Init, QualType Target) -> SmallVector<NarrowingInfo> {
    if (!Init)
      return {};
    if (auto Info = isScalarNarrowing(Init, Target, Ctx))
      return {*Info};
    return {};
  };

  MatchAnalysis Res;
  if (const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor")) {
    // A class-type variable declaration binds both 'var' and the nested
    // 'ctor', which carries the parentheses and arguments we need.
    const SourceRange Parens = Ctor->getParenOrBraceRange();
    Res.Range =
        ParenRange{Ctor->getBeginLoc(), Parens.getBegin(), Parens.getEnd()};
    Res.Narrowings = isCtorNarrowing(Ctor, Ctx);
  } else if (const auto *PLE =
                 Result.Nodes.getNodeAs<CXXParenListInitExpr>("ple")) {
    Res.Range = handlePLE(PLE, Result);
    Res.Narrowings = isPLENarrowing(PLE, Ctx);
  } else if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var")) {
    Res.Range = handleScalarVar(Var, SM, LangOpts);
    Res.Narrowings = ScalarNarrowing(Var->getInit(), Var->getType());
  } else {
    llvm_unreachable("No matches found");
  }
  return Res;
}

void UseBracedInitializationCheck::check(
    const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = Result.Context->getLangOpts();
  const ASTContext &Ctx = *Result.Context;

  const auto [Range, Narrowings] = analyzeMatch(Result, Ctx, SM, LangOpts);

  if (!Range || Range->LParen.isMacroID() || Range->RParen.isMacroID())
    return;

  if (Narrowings.empty()) {
    diag(Range->DiagLoc, "use braced initialization instead of parenthesized "
                         "initialization")
        << FixItHint::CreateReplacement(Range->LParen, "{")
        << FixItHint::CreateReplacement(Range->RParen, "}");
    return;
  }

  diag(Range->DiagLoc, "use braced initialization instead of parenthesized "
                       "initialization");
  for (const auto &[I, N] : llvm::enumerate(Narrowings)) {
    auto Note = diag(N.Loc,
                     "narrowing conversion from %0 to %1 will be ill-formed in "
                     "braced initialization",
                     DiagnosticIDs::Note);
    Note << N.From << N.To;
    if (I == 0)
      Note << FixItHint::CreateReplacement(Range->LParen, "{")
           << FixItHint::CreateReplacement(Range->RParen, "}");
  }
}

} // namespace clang::tidy::misc
