//===--- NotNullTerminatedResultCheck.cpp - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NotNullTerminatedResultCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/PPCallbacks.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

constexpr llvm::StringLiteral FunctionExprName = "FunctionExpr";
constexpr llvm::StringLiteral CastExprName = "CastExpr";
constexpr llvm::StringLiteral UnknownDestName = "UnknownDest";
constexpr llvm::StringLiteral DestArrayTyName = "DestArrayTy";
constexpr llvm::StringLiteral DestVarDeclName = "DestVarDecl";
constexpr llvm::StringLiteral DestMallocExprName = "DestMalloc";
constexpr llvm::StringLiteral DestExprName = "DestExpr";
constexpr llvm::StringLiteral SrcVarDeclName = "SrcVarDecl";
constexpr llvm::StringLiteral SrcExprName = "SrcExpr";
constexpr llvm::StringLiteral LengthExprName = "LengthExpr";
constexpr llvm::StringLiteral WrongLengthExprName = "WrongLength";
constexpr llvm::StringLiteral UnknownLengthName = "UnknownLength";

enum class LengthHandleKind { Increase, Decrease };

namespace {
static Preprocessor *PP;
} // namespace

// Returns the expression of destination's capacity which is part of a
// 'VariableArrayType', 'ConstantArrayTypeLoc' or an argument of a 'malloc()'
// family function call.
static const Expr *getDestCapacityExpr(const MatchFinder::MatchResult &Result) {
  if (const auto *DestMalloc = Result.Nodes.getNodeAs<Expr>(DestMallocExprName))
    return DestMalloc;

  if (const auto *DestVAT =
          Result.Nodes.getNodeAs<VariableArrayType>(DestArrayTyName))
    return DestVAT->getSizeExpr();

  if (const auto *DestVD = Result.Nodes.getNodeAs<VarDecl>(DestVarDeclName))
    if (const TypeLoc DestTL = DestVD->getTypeSourceInfo()->getTypeLoc())
      if (const auto DestCTL = DestTL.getAs<ConstantArrayTypeLoc>())
        return DestCTL.getSizeExpr();

  return nullptr;
}

// Returns the length of \p E as an 'IntegerLiteral' or a 'StringLiteral'
// without the null-terminator.
static unsigned getLength(const Expr *E,
                          const MatchFinder::MatchResult &Result) {
  if (!E)
    return 0;

  Expr::EvalResult Length;
  E = E->IgnoreImpCasts();

  if (const auto *LengthDRE = dyn_cast<DeclRefExpr>(E))
    if (const auto *LengthVD = dyn_cast<VarDecl>(LengthDRE->getDecl()))
      if (!isa<ParmVarDecl>(LengthVD))
        if (const Expr *LengthInit = LengthVD->getInit())
          if (LengthInit->EvaluateAsInt(Length, *Result.Context))
            return Length.Val.getInt().getZExtValue();

  if (const auto *LengthIL = dyn_cast<IntegerLiteral>(E))
    return LengthIL->getValue().getZExtValue();

  if (const auto *StrDRE = dyn_cast<DeclRefExpr>(E))
    if (const auto *StrVD = dyn_cast<VarDecl>(StrDRE->getDecl()))
      if (const Expr *StrInit = StrVD->getInit())
        if (const auto *StrSL =
                dyn_cast<StringLiteral>(StrInit->IgnoreImpCasts()))
          return StrSL->getLength();

  if (const auto *SrcSL = dyn_cast<StringLiteral>(E))
    return SrcSL->getLength();

  return 0;
}

// Returns the capacity of the destination array.
// For example in 'char dest[13]; memcpy(dest, ...)' it returns 13.
static int getDestCapacity(const MatchFinder::MatchResult &Result) {
  if (const auto *DestCapacityExpr = getDestCapacityExpr(Result))
    return getLength(DestCapacityExpr, Result);

  return 0;
}

// Returns the 'strlen()' if it is the given length.
static const CallExpr *getStrlenExpr(const MatchFinder::MatchResult &Result) {
  if (const auto *StrlenExpr =
          Result.Nodes.getNodeAs<CallExpr>(WrongLengthExprName))
    if (const Decl *D = StrlenExpr->getCalleeDecl())
      if (const FunctionDecl *FD = D->getAsFunction())
        if (const IdentifierInfo *II = FD->getIdentifier())
          if (II->isStr("strlen") || II->isStr("wcslen"))
            return StrlenExpr;

  return nullptr;
}

// Returns the length which is given in the memory/string handler function.
// For example in 'memcpy(dest, "foobar", 3)' it returns 3.
static int getGivenLength(const MatchFinder::MatchResult &Result) {
  if (Result.Nodes.getNodeAs<Expr>(UnknownLengthName))
    return 0;

  if (int Length =
          getLength(Result.Nodes.getNodeAs<Expr>(WrongLengthExprName), Result))
    return Length;

  if (int Length =
          getLength(Result.Nodes.getNodeAs<Expr>(LengthExprName), Result))
    return Length;

  // Special case, for example 'strlen("foo")'.
  if (const CallExpr *StrlenCE = getStrlenExpr(Result))
    if (const Expr *Arg = StrlenCE->getArg(0)->IgnoreImpCasts())
      if (int ArgLength = getLength(Arg, Result))
        return ArgLength;

  return 0;
}

// Returns a string representation of \p E.
static StringRef exprToStr(const Expr *E,
                           const MatchFinder::MatchResult &Result) {
  if (!E)
    return "";

  return Lexer::getSourceText(
      CharSourceRange::getTokenRange(E->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts(), 0);
}

// Returns the proper token based end location of \p E.
static SourceLocation exprLocEnd(const Expr *E,
                                 const MatchFinder::MatchResult &Result) {
  return Lexer::getLocForEndOfToken(E->getEndLoc(), 0, *Result.SourceManager,
                                    Result.Context->getLangOpts());
}

//===----------------------------------------------------------------------===//
// Rewrite decision helper functions.
//===----------------------------------------------------------------------===//

// Increment by integer '1' can result in overflow if it is the maximal value.
// After that it would be extended to 'size_t' and its value would be wrong,
// therefore we have to inject '+ 1UL' instead.
static bool isInjectUL(const MatchFinder::MatchResult &Result) {
  return getGivenLength(Result) == std::numeric_limits<int>::max();
}

// If the capacity of the destination array is unknown it is denoted as unknown.
static bool isKnownDest(const MatchFinder::MatchResult &Result) {
  return !Result.Nodes.getNodeAs<Expr>(UnknownDestName);
}

// True if the capacity of the destination array is based on the given length,
// therefore we assume that it cannot overflow (e.g. 'malloc(given_length + 1)'
static bool isDestBasedOnGivenLength(const MatchFinder::MatchResult &Result) {
  StringRef DestCapacityExprStr =
      exprToStr(getDestCapacityExpr(Result), Result).trim();
  StringRef LengthExprStr =
      exprToStr(Result.Nodes.getNodeAs<Expr>(LengthExprName), Result).trim();

  return DestCapacityExprStr != "" && LengthExprStr != "" &&
         DestCapacityExprStr.contains(LengthExprStr);
}

// Writing and reading from the same memory cannot remove the null-terminator.
static bool isDestAndSrcEquals(const MatchFinder::MatchResult &Result) {
  if (const auto *DestDRE = Result.Nodes.getNodeAs<DeclRefExpr>(DestExprName))
    if (const auto *SrcDRE = Result.Nodes.getNodeAs<DeclRefExpr>(SrcExprName))
      return DestDRE->getDecl()->getCanonicalDecl() ==
             SrcDRE->getDecl()->getCanonicalDecl();

  return false;
}

// For example 'std::string str = "foo"; memcpy(dst, str.data(), str.length())'.
static bool isStringDataAndLength(const MatchFinder::MatchResult &Result) {
  const auto *DestExpr =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>(DestExprName);
  const auto *SrcExpr = Result.Nodes.getNodeAs<CXXMemberCallExpr>(SrcExprName);
  const auto *LengthExpr =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>(WrongLengthExprName);

  StringRef DestStr = "", SrcStr = "", LengthStr = "";
  if (DestExpr)
    if (const CXXMethodDecl *DestMD = DestExpr->getMethodDecl())
      DestStr = DestMD->getName();

  if (SrcExpr)
    if (const CXXMethodDecl *SrcMD = SrcExpr->getMethodDecl())
      SrcStr = SrcMD->getName();

  if (LengthExpr)
    if (const CXXMethodDecl *LengthMD = LengthExpr->getMethodDecl())
      LengthStr = LengthMD->getName();

  return (LengthStr == "length" || LengthStr == "size") &&
         (SrcStr == "data" || DestStr == "data");
}

static bool
isGivenLengthEqualToSrcLength(const MatchFinder::MatchResult &Result) {
  if (Result.Nodes.getNodeAs<Expr>(UnknownLengthName))
    return false;

  if (isStringDataAndLength(Result))
    return true;

  int GivenLength = getGivenLength(Result);
  int SrcLength = getLength(Result.Nodes.getNodeAs<Expr>(SrcExprName), Result);

  if (GivenLength != 0 && SrcLength != 0 && GivenLength == SrcLength)
    return true;

  if (const auto *LengthExpr = Result.Nodes.getNodeAs<Expr>(LengthExprName))
    if (dyn_cast<BinaryOperator>(LengthExpr->IgnoreParenImpCasts()))
      return false;

  // Check the strlen()'s argument's 'VarDecl' is equal to the source 'VarDecl'.
  if (const CallExpr *StrlenCE = getStrlenExpr(Result))
    if (const auto *ArgDRE =
            dyn_cast<DeclRefExpr>(StrlenCE->getArg(0)->IgnoreImpCasts()))
      if (const auto *SrcVD = Result.Nodes.getNodeAs<VarDecl>(SrcVarDeclName))
        return dyn_cast<VarDecl>(ArgDRE->getDecl()) == SrcVD;

  return false;
}

static bool isCorrectGivenLength(const MatchFinder::MatchResult &Result) {
  if (Result.Nodes.getNodeAs<Expr>(UnknownLengthName))
    return false;

  return !isGivenLengthEqualToSrcLength(Result);
}

// If we rewrite the function call we need to create extra space to hold the
// null terminator. The new necessary capacity overflows without that '+ 1'
// size and we need to correct the given capacity.
static bool isDestCapacityOverflows(const MatchFinder::MatchResult &Result) {
  if (!isKnownDest(Result))
    return true;

  const Expr *DestCapacityExpr = getDestCapacityExpr(Result);
  int DestCapacity = getLength(DestCapacityExpr, Result);
  int GivenLength = getGivenLength(Result);

  if (GivenLength != 0 && DestCapacity != 0)
    return isGivenLengthEqualToSrcLength(Result) && DestCapacity == GivenLength;

  // Assume that the destination array's capacity cannot overflow if the
  // expression of the memory allocation contains '+ 1'.
  StringRef DestCapacityExprStr = exprToStr(DestCapacityExpr, Result);
  if (DestCapacityExprStr.contains("+1") || DestCapacityExprStr.contains("+ 1"))
    return false;

  return true;
}

static bool
isFixedGivenLengthAndUnknownSrc(const MatchFinder::MatchResult &Result) {
  if (Result.Nodes.getNodeAs<IntegerLiteral>(WrongLengthExprName))
    return !getLength(Result.Nodes.getNodeAs<Expr>(SrcExprName), Result);

  return false;
}

//===----------------------------------------------------------------------===//
// Code injection functions.
//===----------------------------------------------------------------------===//

// Increase or decrease \p LengthExpr by one.
static void lengthExprHandle(const Expr *LengthExpr,
                             LengthHandleKind LengthHandle,
                             const MatchFinder::MatchResult &Result,
                             DiagnosticBuilder &Diag) {
  LengthExpr = LengthExpr->IgnoreParenImpCasts();

  // See whether we work with a macro.
  bool IsMacroDefinition = false;
  StringRef LengthExprStr = exprToStr(LengthExpr, Result);
  Preprocessor::macro_iterator It = PP->macro_begin();
  while (It != PP->macro_end() && !IsMacroDefinition) {
    if (It->first->getName() == LengthExprStr)
      IsMacroDefinition = true;

    ++It;
  }

  // Try to obtain an 'IntegerLiteral' and adjust it.
  if (!IsMacroDefinition) {
    if (const auto *LengthIL = dyn_cast<IntegerLiteral>(LengthExpr)) {
      size_t NewLength = LengthIL->getValue().getZExtValue() +
                         (LengthHandle == LengthHandleKind::Increase
                              ? (isInjectUL(Result) ? 1UL : 1)
                              : -1);

      const auto NewLengthFix = FixItHint::CreateReplacement(
          LengthIL->getSourceRange(),
          (Twine(NewLength) + (isInjectUL(Result) ? "UL" : "")).str());
      Diag << NewLengthFix;
      return;
    }
  }

  // Try to obtain and remove the '+ 1' string as a decrement fix.
  const auto *BO = dyn_cast<BinaryOperator>(LengthExpr);
  if (BO && BO->getOpcode() == BO_Add &&
      LengthHandle == LengthHandleKind::Decrease) {
    const Expr *LhsExpr = BO->getLHS()->IgnoreImpCasts();
    const Expr *RhsExpr = BO->getRHS()->IgnoreImpCasts();

    if (const auto *LhsIL = dyn_cast<IntegerLiteral>(LhsExpr)) {
      if (LhsIL->getValue().getZExtValue() == 1) {
        Diag << FixItHint::CreateRemoval(
            {LhsIL->getBeginLoc(),
             RhsExpr->getBeginLoc().getLocWithOffset(-1)});
        return;
      }
    }

    if (const auto *RhsIL = dyn_cast<IntegerLiteral>(RhsExpr)) {
      if (RhsIL->getValue().getZExtValue() == 1) {
        Diag << FixItHint::CreateRemoval(
            {LhsExpr->getEndLoc().getLocWithOffset(1), RhsIL->getEndLoc()});
        return;
      }
    }
  }

  // Try to inject the '+ 1'/'- 1' string.
  bool NeedInnerParen = BO && BO->getOpcode() != BO_Add;

  if (NeedInnerParen)
    Diag << FixItHint::CreateInsertion(LengthExpr->getBeginLoc(), "(");

  SmallString<8> Injection;
  if (NeedInnerParen)
    Injection += ')';
  Injection += LengthHandle == LengthHandleKind::Increase ? " + 1" : " - 1";
  if (isInjectUL(Result))
    Injection += "UL";

  Diag << FixItHint::CreateInsertion(exprLocEnd(LengthExpr, Result), Injection);
}

static void lengthArgHandle(LengthHandleKind LengthHandle,
                            const MatchFinder::MatchResult &Result,
                            DiagnosticBuilder &Diag) {
  const auto *LengthExpr = Result.Nodes.getNodeAs<Expr>(LengthExprName);
  lengthExprHandle(LengthExpr, LengthHandle, Result, Diag);
}

static void lengthArgPosHandle(unsigned ArgPos, LengthHandleKind LengthHandle,
                               const MatchFinder::MatchResult &Result,
                               DiagnosticBuilder &Diag) {
  const auto *FunctionExpr = Result.Nodes.getNodeAs<CallExpr>(FunctionExprName);
  lengthExprHandle(FunctionExpr->getArg(ArgPos), LengthHandle, Result, Diag);
}

// The string handler functions are only operates with plain 'char'/'wchar_t'
// without 'unsigned/signed', therefore we need to cast it.
static bool isDestExprFix(const MatchFinder::MatchResult &Result,
                          DiagnosticBuilder &Diag) {
  const auto *Dest = Result.Nodes.getNodeAs<Expr>(DestExprName);
  if (!Dest)
    return false;

  std::string TempTyStr = Dest->getType().getAsString();
  StringRef TyStr = TempTyStr;
  if (TyStr.startswith("char") || TyStr.startswith("wchar_t"))
    return false;

  Diag << FixItHint::CreateInsertion(Dest->getBeginLoc(), "(char *)");
  return true;
}

// If the destination array is the same length as the given length we have to
// increase the capacity by one to create space for the the null terminator.
static bool isDestCapacityFix(const MatchFinder::MatchResult &Result,
                              DiagnosticBuilder &Diag) {
  bool IsOverflows = isDestCapacityOverflows(Result);
  if (IsOverflows)
    if (const Expr *CapacityExpr = getDestCapacityExpr(Result))
      lengthExprHandle(CapacityExpr, LengthHandleKind::Increase, Result, Diag);

  return IsOverflows;
}

static void removeArg(int ArgPos, const MatchFinder::MatchResult &Result,
                      DiagnosticBuilder &Diag) {
  // This is the following structure: (src, '\0', strlen(src))
  //                     ArgToRemove:             ~~~~~~~~~~~
  //                          LHSArg:       ~~~~
  //                    RemoveArgFix:           ~~~~~~~~~~~~~
  const auto *FunctionExpr = Result.Nodes.getNodeAs<CallExpr>(FunctionExprName);
  const Expr *ArgToRemove = FunctionExpr->getArg(ArgPos);
  const Expr *LHSArg = FunctionExpr->getArg(ArgPos - 1);
  const auto RemoveArgFix = FixItHint::CreateRemoval(
      SourceRange(exprLocEnd(LHSArg, Result),
                  exprLocEnd(ArgToRemove, Result).getLocWithOffset(-1)));
  Diag << RemoveArgFix;
}

static void renameFunc(StringRef NewFuncName,
                       const MatchFinder::MatchResult &Result,
                       DiagnosticBuilder &Diag) {
  const auto *FunctionExpr = Result.Nodes.getNodeAs<CallExpr>(FunctionExprName);
  int FuncNameLength =
      FunctionExpr->getDirectCallee()->getIdentifier()->getLength();
  SourceRange FuncNameRange(
      FunctionExpr->getBeginLoc(),
      FunctionExpr->getBeginLoc().getLocWithOffset(FuncNameLength - 1));

  const auto FuncNameFix =
      FixItHint::CreateReplacement(FuncNameRange, NewFuncName);
  Diag << FuncNameFix;
}

static void renameMemcpy(StringRef Name, bool IsCopy, bool IsSafe,
                         const MatchFinder::MatchResult &Result,
                         DiagnosticBuilder &Diag) {
  SmallString<10> NewFuncName;
  NewFuncName = (Name[0] != 'w') ? "str" : "wcs";
  NewFuncName += IsCopy ? "cpy" : "ncpy";
  NewFuncName += IsSafe ? "_s" : "";
  renameFunc(NewFuncName, Result, Diag);
}

static void insertDestCapacityArg(bool IsOverflows, StringRef Name,
                                  const MatchFinder::MatchResult &Result,
                                  DiagnosticBuilder &Diag) {
  const auto *FunctionExpr = Result.Nodes.getNodeAs<CallExpr>(FunctionExprName);
  SmallString<64> NewSecondArg;

  if (int DestLength = getDestCapacity(Result)) {
    NewSecondArg = Twine(IsOverflows ? DestLength + 1 : DestLength).str();
  } else {
    NewSecondArg =
        (Twine(exprToStr(getDestCapacityExpr(Result), Result)) +
         (IsOverflows ? (!isInjectUL(Result) ? " + 1" : " + 1UL") : ""))
            .str();
  }

  NewSecondArg += ", ";
  const auto InsertNewArgFix = FixItHint::CreateInsertion(
      FunctionExpr->getArg(1)->getBeginLoc(), NewSecondArg);
  Diag << InsertNewArgFix;
}

static void insertNullTerminatorExpr(StringRef Name,
                                     const MatchFinder::MatchResult &Result,
                                     DiagnosticBuilder &Diag) {
  const auto *FunctionExpr = Result.Nodes.getNodeAs<CallExpr>(FunctionExprName);
  int FuncLocStartColumn = Result.SourceManager->getPresumedColumnNumber(
      FunctionExpr->getBeginLoc());
  SourceRange SpaceRange(
      FunctionExpr->getBeginLoc().getLocWithOffset(-FuncLocStartColumn + 1),
      FunctionExpr->getBeginLoc());
  StringRef SpaceBeforeStmtStr = Lexer::getSourceText(
      CharSourceRange::getCharRange(SpaceRange), *Result.SourceManager,
      Result.Context->getLangOpts(), 0);

  SmallString<128> NewAddNullTermExprStr;
  NewAddNullTermExprStr =
      (Twine('\n') + SpaceBeforeStmtStr +
       exprToStr(Result.Nodes.getNodeAs<Expr>(DestExprName), Result) + "[" +
       exprToStr(Result.Nodes.getNodeAs<Expr>(LengthExprName), Result) +
       "] = " + ((Name[0] != 'w') ? "\'\\0\';" : "L\'\\0\';"))
          .str();

  const auto AddNullTerminatorExprFix = FixItHint::CreateInsertion(
      exprLocEnd(FunctionExpr, Result).getLocWithOffset(1),
      NewAddNullTermExprStr);
  Diag << AddNullTerminatorExprFix;
}

//===----------------------------------------------------------------------===//
// Checker logic with the matchers.
//===----------------------------------------------------------------------===//

NotNullTerminatedResultCheck::NotNullTerminatedResultCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WantToUseSafeFunctions(Options.get("WantToUseSafeFunctions", 1)) {}

void NotNullTerminatedResultCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WantToUseSafeFunctions", WantToUseSafeFunctions);
}

void NotNullTerminatedResultCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *pp, Preprocessor *ModuleExpanderPP) {
  PP = pp;
}

namespace {
AST_MATCHER_P(Expr, hasDefinition, ast_matchers::internal::Matcher<Expr>,
              InnerMatcher) {
  const Expr *SimpleNode = &Node;
  SimpleNode = SimpleNode->IgnoreParenImpCasts();

  if (InnerMatcher.matches(*SimpleNode, Finder, Builder))
    return true;

  auto DREHasInit = ignoringImpCasts(
      declRefExpr(to(varDecl(hasInitializer(ignoringImpCasts(InnerMatcher))))));

  if (DREHasInit.matches(*SimpleNode, Finder, Builder))
    return true;

  const char *const VarDeclName = "variable-declaration";
  auto DREHasDefinition = ignoringImpCasts(declRefExpr(
      allOf(to(varDecl().bind(VarDeclName)),
            hasAncestor(compoundStmt(hasDescendant(binaryOperator(
                hasLHS(declRefExpr(to(varDecl(equalsBoundNode(VarDeclName))))),
                hasRHS(ignoringImpCasts(InnerMatcher)))))))));

  if (DREHasDefinition.matches(*SimpleNode, Finder, Builder))
    return true;

  return false;
}
} // namespace

void NotNullTerminatedResultCheck::registerMatchers(MatchFinder *Finder) {
  auto IncOp =
      binaryOperator(hasOperatorName("+"),
                     hasEitherOperand(ignoringParenImpCasts(integerLiteral())));

  auto DecOp =
      binaryOperator(hasOperatorName("-"),
                     hasEitherOperand(ignoringParenImpCasts(integerLiteral())));

  auto HasIncOp = anyOf(ignoringImpCasts(IncOp), hasDescendant(IncOp));
  auto HasDecOp = anyOf(ignoringImpCasts(DecOp), hasDescendant(DecOp));

  auto Container = ignoringImpCasts(cxxMemberCallExpr(hasDescendant(declRefExpr(
      hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(recordDecl(
          hasAnyName("::std::vector", "::std::list", "::std::deque"))))))))));

  auto StringTy = type(hasUnqualifiedDesugaredType(recordType(
      hasDeclaration(cxxRecordDecl(hasName("::std::basic_string"))))));

  auto AnyOfStringTy =
      anyOf(hasType(StringTy), hasType(qualType(pointsTo(StringTy))));

  auto CharTyArray = hasType(qualType(hasCanonicalType(
      arrayType(hasElementType(isAnyCharacter())).bind(DestArrayTyName))));

  auto CharTyPointer = hasType(
      qualType(hasCanonicalType(pointerType(pointee(isAnyCharacter())))));

  auto AnyOfCharTy = anyOf(CharTyArray, CharTyPointer);

  //===--------------------------------------------------------------------===//
  // The following six cases match problematic length expressions.
  //===--------------------------------------------------------------------===//

  // - Example:  char src[] = "foo";       strlen(src);
  auto Strlen =
      callExpr(callee(functionDecl(hasAnyName("::strlen", "::wcslen"))))
          .bind(WrongLengthExprName);

  // - Example:  std::string str = "foo";  str.size();
  auto SizeOrLength =
      cxxMemberCallExpr(
          allOf(on(expr(AnyOfStringTy).bind("Foo")),
                has(memberExpr(member(hasAnyName("size", "length"))))))
          .bind(WrongLengthExprName);

  // - Example:  char src[] = "foo";       sizeof(src);
  auto SizeOfCharExpr = unaryExprOrTypeTraitExpr(has(expr(AnyOfCharTy)));

  auto WrongLength =
      ignoringImpCasts(anyOf(Strlen, SizeOrLength, hasDescendant(Strlen),
                             hasDescendant(SizeOrLength)));

  // - Example:  length = strlen(src);
  auto DREWithoutInc =
      ignoringImpCasts(declRefExpr(to(varDecl(hasInitializer(WrongLength)))));

  auto AnyOfCallOrDREWithoutInc = anyOf(DREWithoutInc, WrongLength);

  // - Example:  int getLength(const char *str) { return strlen(str); }
  auto CallExprReturnWithoutInc = ignoringImpCasts(callExpr(callee(functionDecl(
      hasBody(has(returnStmt(hasReturnValue(AnyOfCallOrDREWithoutInc))))))));

  // - Example:  int length = getLength(src);
  auto DREHasReturnWithoutInc = ignoringImpCasts(
      declRefExpr(to(varDecl(hasInitializer(CallExprReturnWithoutInc)))));

  auto AnyOfWrongLengthInit =
      anyOf(WrongLength, AnyOfCallOrDREWithoutInc, CallExprReturnWithoutInc,
            DREHasReturnWithoutInc);

  //===--------------------------------------------------------------------===//
  // The following five cases match the 'destination' array length's
  // expression which is used in 'memcpy()' and 'memmove()' matchers.
  //===--------------------------------------------------------------------===//

  // Note: Sometimes the size of char is explicitly written out.
  auto SizeExpr = anyOf(SizeOfCharExpr, integerLiteral(equals(1)));

  auto MallocLengthExpr = allOf(
      callee(functionDecl(
          hasAnyName("::alloca", "::calloc", "malloc", "realloc"))),
      hasAnyArgument(allOf(unless(SizeExpr), expr().bind(DestMallocExprName))));

  // - Example:  (char *)malloc(length);
  auto DestMalloc = anyOf(callExpr(MallocLengthExpr),
                          hasDescendant(callExpr(MallocLengthExpr)));

  // - Example:  new char[length];
  auto DestCXXNewExpr = ignoringImpCasts(
      cxxNewExpr(hasArraySize(expr().bind(DestMallocExprName))));

  auto AnyOfDestInit = anyOf(DestMalloc, DestCXXNewExpr);

  // - Example:  char dest[13];  or  char dest[length];
  auto DestArrayTyDecl = declRefExpr(
      to(anyOf(varDecl(CharTyArray).bind(DestVarDeclName),
               varDecl(hasInitializer(AnyOfDestInit)).bind(DestVarDeclName))));

  // - Example:  foo[bar[baz]].qux; (or just ParmVarDecl)
  auto DestUnknownDecl =
      declRefExpr(allOf(to(varDecl(AnyOfCharTy).bind(DestVarDeclName)),
                        expr().bind(UnknownDestName)))
          .bind(DestExprName);

  auto AnyOfDestDecl = ignoringImpCasts(
      anyOf(allOf(hasDefinition(anyOf(AnyOfDestInit, DestArrayTyDecl,
                                      hasDescendant(DestArrayTyDecl))),
                  expr().bind(DestExprName)),
            anyOf(DestUnknownDecl, hasDescendant(DestUnknownDecl))));

  auto NullTerminatorExpr = binaryOperator(
      hasLHS(anyOf(hasDescendant(declRefExpr(
                       to(varDecl(equalsBoundNode(DestVarDeclName))))),
                   hasDescendant(declRefExpr(equalsBoundNode(DestExprName))))),
      hasRHS(ignoringImpCasts(
          anyOf(characterLiteral(equals(0U)), integerLiteral(equals(0))))));

  auto SrcDecl = declRefExpr(
      allOf(to(decl().bind(SrcVarDeclName)),
            anyOf(hasAncestor(cxxMemberCallExpr().bind(SrcExprName)),
                  expr().bind(SrcExprName))));

  auto AnyOfSrcDecl =
      ignoringImpCasts(anyOf(stringLiteral().bind(SrcExprName),
                             hasDescendant(stringLiteral().bind(SrcExprName)),
                             SrcDecl, hasDescendant(SrcDecl)));

  //===--------------------------------------------------------------------===//
  // Match the problematic function calls.
  //===--------------------------------------------------------------------===//

  struct CallContext {
    CallContext(StringRef Name, Optional<unsigned> DestinationPos,
                Optional<unsigned> SourcePos, unsigned LengthPos,
                bool WithIncrease)
        : Name(Name), DestinationPos(DestinationPos), SourcePos(SourcePos),
          LengthPos(LengthPos), WithIncrease(WithIncrease){};

    StringRef Name;
    Optional<unsigned> DestinationPos;
    Optional<unsigned> SourcePos;
    unsigned LengthPos;
    bool WithIncrease;
  };

  auto MatchDestination = [=](CallContext CC) {
    return hasArgument(*CC.DestinationPos,
                       allOf(AnyOfDestDecl,
                             unless(hasAncestor(compoundStmt(
                                 hasDescendant(NullTerminatorExpr)))),
                             unless(Container)));
  };

  auto MatchSource = [=](CallContext CC) {
    return hasArgument(*CC.SourcePos, AnyOfSrcDecl);
  };

  auto MatchGivenLength = [=](CallContext CC) {
    return hasArgument(
        CC.LengthPos,
        allOf(
            anyOf(
                ignoringImpCasts(integerLiteral().bind(WrongLengthExprName)),
                allOf(unless(hasDefinition(SizeOfCharExpr)),
                      allOf(CC.WithIncrease
                                ? ignoringImpCasts(hasDefinition(HasIncOp))
                                : ignoringImpCasts(allOf(
                                      unless(hasDefinition(HasIncOp)),
                                      anyOf(hasDefinition(binaryOperator().bind(
                                                UnknownLengthName)),
                                            hasDefinition(anything())))),
                            AnyOfWrongLengthInit))),
            expr().bind(LengthExprName)));
  };

  auto MatchCall = [=](CallContext CC) {
    std::string CharHandlerFuncName = "::" + CC.Name.str();

    // Try to match with 'wchar_t' based function calls.
    std::string WcharHandlerFuncName =
        "::" + (CC.Name.startswith("mem") ? "w" + CC.Name.str()
                                          : "wcs" + CC.Name.substr(3).str());

    return allOf(callee(functionDecl(
                     hasAnyName(CharHandlerFuncName, WcharHandlerFuncName))),
                 MatchGivenLength(CC));
  };

  auto Match = [=](CallContext CC) {
    if (CC.DestinationPos && CC.SourcePos)
      return allOf(MatchCall(CC), MatchDestination(CC), MatchSource(CC));

    if (CC.DestinationPos && !CC.SourcePos)
      return allOf(MatchCall(CC), MatchDestination(CC),
                   hasArgument(*CC.DestinationPos, anything()));

    if (!CC.DestinationPos && CC.SourcePos)
      return allOf(MatchCall(CC), MatchSource(CC),
                   hasArgument(*CC.SourcePos, anything()));

    llvm_unreachable("Unhandled match");
  };

  // void *memcpy(void *dest, const void *src, size_t count)
  auto Memcpy = Match({"memcpy", 0, 1, 2, false});

  // errno_t memcpy_s(void *dest, size_t ds, const void *src, size_t count)
  auto Memcpy_s = Match({"memcpy_s", 0, 2, 3, false});

  // void *memchr(const void *src, int c, size_t count)
  auto Memchr = Match({"memchr", None, 0, 2, false});

  // void *memmove(void *dest, const void *src, size_t count)
  auto Memmove = Match({"memmove", 0, 1, 2, false});

  // errno_t memmove_s(void *dest, size_t ds, const void *src, size_t count)
  auto Memmove_s = Match({"memmove_s", 0, 2, 3, false});

  // int strncmp(const char *str1, const char *str2, size_t count);
  auto StrncmpRHS = Match({"strncmp", None, 1, 2, true});
  auto StrncmpLHS = Match({"strncmp", None, 0, 2, true});

  // size_t strxfrm(char *dest, const char *src, size_t count);
  auto Strxfrm = Match({"strxfrm", 0, 1, 2, false});

  // errno_t strerror_s(char *buffer, size_t bufferSize, int errnum);
  auto Strerror_s = Match({"strerror_s", 0, None, 1, false});

  auto AnyOfMatchers = anyOf(Memcpy, Memcpy_s, Memmove, Memmove_s, StrncmpRHS,
                             StrncmpLHS, Strxfrm, Strerror_s);

  Finder->addMatcher(callExpr(AnyOfMatchers).bind(FunctionExprName), this);

  // Need to remove the CastExpr from 'memchr()' as 'strchr()' returns 'char *'.
  Finder->addMatcher(
      callExpr(Memchr,
               unless(hasAncestor(castExpr(unless(implicitCastExpr())))))
          .bind(FunctionExprName),
      this);
  Finder->addMatcher(
      castExpr(allOf(unless(implicitCastExpr()),
                     has(callExpr(Memchr).bind(FunctionExprName))))
          .bind(CastExprName),
      this);
}

void NotNullTerminatedResultCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FunctionExpr = Result.Nodes.getNodeAs<CallExpr>(FunctionExprName);
  if (FunctionExpr->getBeginLoc().isMacroID())
    return;

  if (WantToUseSafeFunctions && PP->isMacroDefined("__STDC_LIB_EXT1__")) {
    Optional<bool> AreSafeFunctionsWanted;

    Preprocessor::macro_iterator It = PP->macro_begin();
    while (It != PP->macro_end() && !AreSafeFunctionsWanted.hasValue()) {
      if (It->first->getName() == "__STDC_WANT_LIB_EXT1__") {
        const auto *MI = PP->getMacroInfo(It->first);
        const auto &T = MI->tokens().back();
        StringRef ValueStr = StringRef(T.getLiteralData(), T.getLength());
        llvm::APInt IntValue;
        ValueStr.getAsInteger(10, IntValue);
        AreSafeFunctionsWanted = IntValue.getZExtValue();
      }

      ++It;
    }

    if (AreSafeFunctionsWanted.hasValue())
      UseSafeFunctions = AreSafeFunctionsWanted.getValue();
  }

  StringRef Name = FunctionExpr->getDirectCallee()->getName();
  if (Name.startswith("mem") || Name.startswith("wmem"))
    memoryHandlerFunctionFix(Name, Result);
  else if (Name == "strerror_s")
    strerror_sFix(Result);
  else if (Name.endswith("ncmp"))
    ncmpFix(Name, Result);
  else if (Name.endswith("xfrm"))
    xfrmFix(Name, Result);
}

void NotNullTerminatedResultCheck::memoryHandlerFunctionFix(
    StringRef Name, const MatchFinder::MatchResult &Result) {
  if (isCorrectGivenLength(Result))
    return;

  if (Name.endswith("chr")) {
    memchrFix(Name, Result);
    return;
  }

  if ((Name.contains("cpy") || Name.contains("move")) &&
      (isDestAndSrcEquals(Result) || isFixedGivenLengthAndUnknownSrc(Result)))
    return;

  auto Diag =
      diag(Result.Nodes.getNodeAs<CallExpr>(FunctionExprName)->getBeginLoc(),
           "the result from calling '%0' is not null-terminated")
      << Name;

  if (Name.endswith("cpy")) {
    memcpyFix(Name, Result, Diag);
  } else if (Name.endswith("cpy_s")) {
    memcpy_sFix(Name, Result, Diag);
  } else if (Name.endswith("move")) {
    memmoveFix(Name, Result, Diag);
  } else if (Name.endswith("move_s")) {
    isDestCapacityFix(Result, Diag);
    lengthArgHandle(LengthHandleKind::Increase, Result, Diag);
  }
}

void NotNullTerminatedResultCheck::memcpyFix(
    StringRef Name, const MatchFinder::MatchResult &Result,
    DiagnosticBuilder &Diag) {
  bool IsOverflows = isDestCapacityFix(Result, Diag);
  bool IsDestFixed = isDestExprFix(Result, Diag);

  bool IsCopy =
      isGivenLengthEqualToSrcLength(Result) || isDestBasedOnGivenLength(Result);

  bool IsSafe = UseSafeFunctions && IsOverflows && isKnownDest(Result) &&
                !isDestBasedOnGivenLength(Result);

  bool IsDestLengthNotRequired =
      IsSafe && getLangOpts().CPlusPlus &&
      Result.Nodes.getNodeAs<ArrayType>(DestArrayTyName) && !IsDestFixed;

  renameMemcpy(Name, IsCopy, IsSafe, Result, Diag);

  if (IsSafe && !IsDestLengthNotRequired)
    insertDestCapacityArg(IsOverflows, Name, Result, Diag);

  if (IsCopy)
    removeArg(2, Result, Diag);

  if (!IsCopy && !IsSafe)
    insertNullTerminatorExpr(Name, Result, Diag);
}

void NotNullTerminatedResultCheck::memcpy_sFix(
    StringRef Name, const MatchFinder::MatchResult &Result,
    DiagnosticBuilder &Diag) {
  bool IsOverflows = isDestCapacityFix(Result, Diag);
  bool IsDestFixed = isDestExprFix(Result, Diag);

  bool RemoveDestLength = getLangOpts().CPlusPlus &&
                          Result.Nodes.getNodeAs<ArrayType>(DestArrayTyName) &&
                          !IsDestFixed;
  bool IsCopy = isGivenLengthEqualToSrcLength(Result);
  bool IsSafe = IsOverflows;

  renameMemcpy(Name, IsCopy, IsSafe, Result, Diag);

  if (!IsSafe || (IsSafe && RemoveDestLength))
    removeArg(1, Result, Diag);
  else if (IsOverflows && isKnownDest(Result))
    lengthArgPosHandle(1, LengthHandleKind::Increase, Result, Diag);

  if (IsCopy)
    removeArg(3, Result, Diag);

  if (!IsCopy && !IsSafe)
    insertNullTerminatorExpr(Name, Result, Diag);
}

void NotNullTerminatedResultCheck::memchrFix(
    StringRef Name, const MatchFinder::MatchResult &Result) {
  const auto *FunctionExpr = Result.Nodes.getNodeAs<CallExpr>(FunctionExprName);
  if (const auto GivenCL = dyn_cast<CharacterLiteral>(FunctionExpr->getArg(1)))
    if (GivenCL->getValue() != 0)
      return;

  auto Diag = diag(FunctionExpr->getArg(2)->IgnoreParenCasts()->getBeginLoc(),
                   "the length is too short to include the null terminator");

  if (const auto *CastExpr = Result.Nodes.getNodeAs<Expr>(CastExprName)) {
    const auto CastRemoveFix = FixItHint::CreateRemoval(
        SourceRange(CastExpr->getBeginLoc(),
                    FunctionExpr->getBeginLoc().getLocWithOffset(-1)));
    Diag << CastRemoveFix;
  }

  StringRef NewFuncName = (Name[0] != 'w') ? "strchr" : "wcschr";
  renameFunc(NewFuncName, Result, Diag);
  removeArg(2, Result, Diag);
}

void NotNullTerminatedResultCheck::memmoveFix(
    StringRef Name, const MatchFinder::MatchResult &Result,
    DiagnosticBuilder &Diag) {
  bool IsOverflows = isDestCapacityFix(Result, Diag);

  if (UseSafeFunctions && isKnownDest(Result)) {
    renameFunc((Name[0] != 'w') ? "memmove_s" : "wmemmove_s", Result, Diag);
    insertDestCapacityArg(IsOverflows, Name, Result, Diag);
  }

  lengthArgHandle(LengthHandleKind::Increase, Result, Diag);
}

void NotNullTerminatedResultCheck::strerror_sFix(
    const MatchFinder::MatchResult &Result) {
  auto Diag =
      diag(Result.Nodes.getNodeAs<CallExpr>(FunctionExprName)->getBeginLoc(),
           "the result from calling 'strerror_s' is not null-terminated and "
           "missing the last character of the error message");

  isDestCapacityFix(Result, Diag);
  lengthArgHandle(LengthHandleKind::Increase, Result, Diag);
}

void NotNullTerminatedResultCheck::ncmpFix(
    StringRef Name, const MatchFinder::MatchResult &Result) {
  const auto *FunctionExpr = Result.Nodes.getNodeAs<CallExpr>(FunctionExprName);
  const Expr *FirstArgExpr = FunctionExpr->getArg(0)->IgnoreImpCasts();
  const Expr *SecondArgExpr = FunctionExpr->getArg(1)->IgnoreImpCasts();
  bool IsLengthTooLong = false;

  if (const CallExpr *StrlenExpr = getStrlenExpr(Result)) {
    const Expr *LengthExprArg = StrlenExpr->getArg(0);
    StringRef FirstExprStr = exprToStr(FirstArgExpr, Result).trim();
    StringRef SecondExprStr = exprToStr(SecondArgExpr, Result).trim();
    StringRef LengthArgStr = exprToStr(LengthExprArg, Result).trim();
    IsLengthTooLong =
        LengthArgStr == FirstExprStr || LengthArgStr == SecondExprStr;
  } else {
    int SrcLength =
        getLength(Result.Nodes.getNodeAs<Expr>(SrcExprName), Result);
    int GivenLength = getGivenLength(Result);
    if (SrcLength != 0 && GivenLength != 0)
      IsLengthTooLong = GivenLength > SrcLength;
  }

  if (!IsLengthTooLong && !isStringDataAndLength(Result))
    return;

  auto Diag = diag(FunctionExpr->getArg(2)->IgnoreParenCasts()->getBeginLoc(),
                   "comparison length is too long and might lead to a "
                   "buffer overflow");

  lengthArgHandle(LengthHandleKind::Decrease, Result, Diag);
}

void NotNullTerminatedResultCheck::xfrmFix(
    StringRef Name, const MatchFinder::MatchResult &Result) {
  if (!isDestCapacityOverflows(Result))
    return;

  auto Diag =
      diag(Result.Nodes.getNodeAs<CallExpr>(FunctionExprName)->getBeginLoc(),
           "the result from calling '%0' is not null-terminated")
      << Name;

  isDestCapacityFix(Result, Diag);
  lengthArgHandle(LengthHandleKind::Increase, Result, Diag);
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
