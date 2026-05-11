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

static SmallVector<const Expr *>
collectExplicitArgs(const CXXConstructExpr &Ctor) {
  SmallVector<const Expr *> ExplicitArgs;
  for (unsigned I = 0; I < Ctor.getNumArgs(); ++I)
    if (!isa<CXXDefaultArgExpr>(Ctor.getArg(I)))
      ExplicitArgs.push_back(Ctor.getArg(I));
  return ExplicitArgs;
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

AST_MATCHER_P(CXXFunctionalCastExpr, hasSubExpr,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  return InnerMatcher.matches(*Node.getSubExpr(), Finder, Builder);
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
  const SmallVector<const Expr *> ExplicitArgs = collectExplicitArgs(Node);
  return hasInitListCtor(RD, ExplicitArgs);
}

AST_POLYMORPHIC_MATCHER(isListInit,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(CXXFunctionalCastExpr,
                                                        CXXConstructExpr)) {
  return Node.isListInitialization();
}

AST_MATCHER(CXXNewExpr, isParenInit) {
  return Node.getInitializationStyle() == CXXNewInitializationStyle::Parens;
}

AST_MATCHER(CXXNewExpr, allocatesAutoType) {
  return Node.getAllocatedTypeSourceInfo()->getType()->getContainedAutoType() !=
         nullptr;
}

AST_MATCHER(CXXNewExpr, allocatesRecordType) {
  return Node.getType()->getPointeeType()->isRecordType();
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

static std::optional<ParenRange>
handleMemInit(const CXXCtorInitializer *MemInit, const CXXConstructExpr *Ctor,
              const SourceManager &SM, const LangOptions &LangOpts) {
  if (!Ctor) {
    // CXXCtorInitializer stores both '(' and '{' locations in the same
    // fields. Only transform parenthesized initialization.
    const SourceLocation LParen = MemInit->getLParenLoc();
    if (!LParen.isValid() || !MemInit->getRParenLoc().isValid())
      return std::nullopt;
    Token Tok;
    if (Lexer::getRawToken(LParen, Tok, SM, LangOpts) ||
        Tok.isNot(tok::l_paren))
      return std::nullopt;
  }
  return ParenRange{MemInit->getSourceLocation(), MemInit->getLParenLoc(),
                    MemInit->getRParenLoc()};
}

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

static std::optional<ParenRange>
handleScalarCast(const CXXFunctionalCastExpr *Cast) {
  const SourceLocation LParen = Cast->getLParenLoc();
  const SourceLocation RParen = Cast->getRParenLoc();
  if (!LParen.isValid() || !RParen.isValid())
    return std::nullopt;
  return ParenRange{Cast->getBeginLoc(), LParen, RParen};
}

static std::optional<ParenRange> handleScalarNew(const CXXNewExpr *New) {
  const SourceRange InitRange = New->getDirectInitRange();
  if (!InitRange.isValid())
    return std::nullopt;
  return ParenRange{New->getBeginLoc(), InitRange.getBegin(),
                    InitRange.getEnd()};
}

static std::optional<ParenRange>
handlePLE(const CXXParenListInitExpr *PLE,
          const MatchFinder::MatchResult &Result) {
  SourceLocation DiagLoc;
  if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var_ple"))
    DiagLoc = Var->getLocation();
  else if (const auto *Cast =
               Result.Nodes.getNodeAs<CXXFunctionalCastExpr>("cast_ple"))
    DiagLoc = Cast->getBeginLoc();
  else if (const auto *New = Result.Nodes.getNodeAs<CXXNewExpr>("new_ple"))
    DiagLoc = New->getBeginLoc();
  else
    llvm_unreachable("No context for CXXParenListInitExpr");
  return ParenRange{DiagLoc, PLE->getBeginLoc(), PLE->getEndLoc()};
}

void UseBracedInitializationCheck::registerMatchers(MatchFinder *Finder) {
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
  Finder->addMatcher(cxxTemporaryObjectExpr(GoodCtor).bind("ctor"), this);
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          cxxFunctionalCastExpr(
              unless(hasType(autoType())), unless(hasType(isDependentType())),
              unless(isInTemplateInstantiation()), unless(isListInit()),
              anyOf(allOf(hasType(hasUnqualifiedDesugaredType(recordType())),
                          hasSubExpr(ignoringImplicit(cxxConstructExpr(
                              unless(isListInit()),
                              unless(canOverlapWithInitListCtor()))))),
                    unless(hasType(hasUnqualifiedDesugaredType(recordType())))))
              .bind("func_cast")),
      this);
  Finder->addMatcher(
      cxxNewExpr(unless(hasType(isDependentType())),
                 anyOf(has(ignoringImplicit(GoodCtorExpr)),
                       allOf(unless(allocatesRecordType()), isParenInit(),
                             unless(allocatesAutoType()))))
          .bind("new_expr"),
      this);
  Finder->addMatcher(
      cxxCtorInitializer(
          isWritten(),
          anyOf(withInitializer(ignoringImplicit(GoodCtorExpr)),
                unless(withInitializer(ignoringImplicit(cxxConstructExpr())))))
          .bind("ctor_init"),
      this);

  if (getLangOpts().CPlusPlus20) {
    auto GoodPLE = CxxParenListInitExpr().bind("ple");
    Finder->addMatcher(varDecl(hasInitStyle(VarDecl::ParenListInit),
                               hasInitializer(GoodPLE), GoodVar)
                           .bind("var_ple"),
                       this);
    Finder->addMatcher(cxxFunctionalCastExpr(has(GoodPLE)).bind("cast_ple"),
                       this);
    Finder->addMatcher(cxxNewExpr(has(GoodPLE)).bind("new_ple"), this);
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
    // A 'ctor' binding can be standalone or nested inside Var/New/MemInit;
    // it always carries the parens and arguments we need.
    const auto *CtorInit =
        Result.Nodes.getNodeAs<CXXCtorInitializer>("ctor_init");
    Res.Range =
        CtorInit
            ? handleMemInit(CtorInit, Ctor, SM, LangOpts)
            : std::optional<ParenRange>(ParenRange{
                  Ctor->getBeginLoc(), Ctor->getParenOrBraceRange().getBegin(),
                  Ctor->getParenOrBraceRange().getEnd()});
    Res.Narrowings = isCtorNarrowing(Ctor, Ctx);
  } else if (const auto *PLE =
                 Result.Nodes.getNodeAs<CXXParenListInitExpr>("ple")) {
    Res.Range = handlePLE(PLE, Result);
    Res.Narrowings = isPLENarrowing(PLE, Ctx);
  } else if (const auto *CtorInit =
                 Result.Nodes.getNodeAs<CXXCtorInitializer>("ctor_init")) {
    Res.Range = handleMemInit(CtorInit, /*Ctor=*/nullptr, SM, LangOpts);
    if (CtorInit->isMemberInitializer())
      Res.Narrowings = ScalarNarrowing(CtorInit->getInit(),
                                       CtorInit->getMember()->getType());
  } else if (const auto *Var = Result.Nodes.getNodeAs<VarDecl>("var")) {
    Res.Range = handleScalarVar(Var, SM, LangOpts);
    Res.Narrowings = ScalarNarrowing(Var->getInit(), Var->getType());
  } else if (const auto *Cast =
                 Result.Nodes.getNodeAs<CXXFunctionalCastExpr>("func_cast")) {
    Res.Range = handleScalarCast(Cast);
    Res.Narrowings = ScalarNarrowing(Cast->getSubExpr(), Cast->getType());
  } else if (const auto *New = Result.Nodes.getNodeAs<CXXNewExpr>("new_expr")) {
    Res.Range = handleScalarNew(New);
    Res.Narrowings =
        ScalarNarrowing(New->getInitializer(), New->getAllocatedType());
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
