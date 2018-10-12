//===--- NotNullTerminatedResultCheck.cpp - clang-tidy ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

static const char *const FuncExprName = "entire-called-function-expr";
static const char *const CastExprName = "cast-expr";
static const char *const UnknownDestName = "destination-length-is-unknown";
static const char *const NotJustCharTyName = "unsigned-or-signed-char";
static const char *const DestArrayTyName = "destination-is-array-type";
static const char *const DestVarDeclName = "destination-variable-declaration";
static const char *const SrcVarDeclName = "source-variable-declaration";
static const char *const UnknownLengthName = "given-length-is-unknown";
static const char *const WrongLengthExprName = "strlen-or-size";
static const char *const DestMallocExprName = "destination-malloc-expr";
static const char *const DestExprName = "destination-decl-ref-expr";
static const char *const SrcExprName = "source-expression-or-string-literal";
static const char *const LengthExprName = "given-length-expression";

enum class LengthHandleKind { Increase, Decrease };

namespace {
static Preprocessor *PP;
} // namespace

// The capacity: VariableArrayType, ConstantArrayType, argument of a 'malloc()'
// family function or an argument of a custom memory allocation.
static const Expr *getDestCapacityExpr(const MatchFinder::MatchResult &Result);

static int getDestCapacity(const MatchFinder::MatchResult &Result);

// Length could be an IntegerLiteral or length of a StringLiteral.
static int getLength(const Expr *E, const MatchFinder::MatchResult &Result);

static int getGivenLength(const MatchFinder::MatchResult &Result);

static StringRef exprToStr(const Expr *E,
                           const MatchFinder::MatchResult &Result);

static SourceLocation exprLocEnd(const Expr *E,
                                 const MatchFinder::MatchResult &Result) {
  return Lexer::getLocForEndOfToken(E->getEndLoc(), 0, *Result.SourceManager,
                                    Result.Context->getLangOpts());
}

//===----------------------------------------------------------------------===//
// Rewrite decision helper functions.
//===----------------------------------------------------------------------===//

// Increment by integer '1' can result in overflow if it is the maximal value.
// After that it will be extended to 'size_t' and its value will be wrong,
// therefore we have to inject '+ 1UL' instead.
static bool isInjectUL(const MatchFinder::MatchResult &Result) {
  return getGivenLength(Result) == std::numeric_limits<int>::max();
}

// If the capacity of the destination array is unknown it is denoted as unknown.
static bool isKnownDest(const MatchFinder::MatchResult &Result) {
  return !Result.Nodes.getNodeAs<Expr>(UnknownDestName);
}

// True if the capacity of the destination array is based on the given length,
// therefore it looks like it cannot overflow (e.g. 'malloc(given_length + 1)'
// Note: If the capacity and the given length is equal then the new function
// is a simple 'cpy()' and because it returns true it prevents increasing the
// given length.
static bool isDestBasedOnGivenLength(const MatchFinder::MatchResult &Result);

// If we write/read from the same array it should be already null-terminated.
static bool isDestAndSrcEquals(const MatchFinder::MatchResult &Result);

// We catch integers as a given length so we have to see if the length of the
// source array is the same length so that the function call is wrong.
static bool isCorrectGivenLength(const MatchFinder::MatchResult &Result);

// Example:  memcpy(dest, str.data(), str.length());
static bool isStringDataAndLength(const MatchFinder::MatchResult &Result);

static bool isDestCapacityOverflows(const MatchFinder::MatchResult &Result);

static bool isLengthEqualToSrcLength(const MatchFinder::MatchResult &Result);

//===----------------------------------------------------------------------===//
// Code injection functions.
//===----------------------------------------------------------------------===//

static void lengthDecrease(const Expr *LengthExpr,
                           const MatchFinder::MatchResult &Result,
                           DiagnosticBuilder &Diag);
static void lengthIncrease(const Expr *LengthExpr,
                           const MatchFinder::MatchResult &Result,
                           DiagnosticBuilder &Diag);

// Increase or decrease an integral expression by one.
static void lengthExprHandle(LengthHandleKind LengthHandle,
                             const Expr *LengthExpr,
                             const MatchFinder::MatchResult &Result,
                             DiagnosticBuilder &Diag);

// Increase or decrease the passed integral argument by one.
static void lengthArgHandle(LengthHandleKind LengthHandle, int ArgPos,
                            const MatchFinder::MatchResult &Result,
                            DiagnosticBuilder &Diag);

// If the destination array is the same length as the given length we have to
// increase the capacity by one to create space for the the null terminator.
static bool destCapacityFix(const MatchFinder::MatchResult &Result,
                            DiagnosticBuilder &Diag);

static void removeArg(int ArgPos, const MatchFinder::MatchResult &Result,
                      DiagnosticBuilder &Diag);

static void renameFunc(StringRef NewFuncName,
                       const MatchFinder::MatchResult &Result,
                       DiagnosticBuilder &Diag);

static void renameMemcpy(StringRef Name, bool IsCopy, bool IsSafe,
                         const MatchFinder::MatchResult &Result,
                         DiagnosticBuilder &Diag);

static void insertDestCapacityArg(bool IsOverflows, StringRef Name,
                                  const MatchFinder::MatchResult &Result,
                                  DiagnosticBuilder &Diag);

static void insertNullTerminatorExpr(StringRef Name,
                                     const MatchFinder::MatchResult &Result,
                                     DiagnosticBuilder &Diag);

NotNullTerminatedResultCheck::NotNullTerminatedResultCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      WantToUseSafeFunctions(Options.get("WantToUseSafeFunctions", 1)) {}

void NotNullTerminatedResultCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "WantToUseSafeFunctions", WantToUseSafeFunctions);
}

void NotNullTerminatedResultCheck::registerPPCallbacks(
    CompilerInstance &Compiler) {
  PP = &Compiler.getPreprocessor();
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

  // - Example:  int getLength(const char *str) { return strlen(str); }
  auto CallExprReturnInit = ignoringImpCasts(
      callExpr(callee(functionDecl(hasBody(has(returnStmt(hasReturnValue(
          ignoringImpCasts(anyOf(DREHasInit, InnerMatcher))))))))));

  if (CallExprReturnInit.matches(*SimpleNode, Finder, Builder))
    return true;

  // - Example:  int length = getLength(src);
  auto DREHasReturnInit = ignoringImpCasts(
      declRefExpr(to(varDecl(hasInitializer(CallExprReturnInit)))));

  if (DREHasReturnInit.matches(*SimpleNode, Finder, Builder))
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

  auto StringTy = type(hasUnqualifiedDesugaredType(recordType(
      hasDeclaration(cxxRecordDecl(hasName("::std::basic_string"))))));

  auto AnyOfStringTy =
      anyOf(hasType(StringTy), hasType(qualType(pointsTo(StringTy))));

  auto CharTy =
      anyOf(asString("char"), asString("wchar_t"),
            allOf(anyOf(asString("unsigned char"), asString("signed char")),
                  type().bind(NotJustCharTyName)));

  auto CharTyArray = hasType(qualType(hasCanonicalType(
      arrayType(hasElementType(CharTy)).bind(DestArrayTyName))));

  auto CharTyPointer =
      hasType(qualType(hasCanonicalType(pointerType(pointee(CharTy)))));

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
          allOf(on(expr(AnyOfStringTy)),
                has(memberExpr(member(hasAnyName("size", "length"))))))
          .bind(WrongLengthExprName);

  // - Example:  char src[] = "foo";       sizeof(src);
  auto SizeOfCharExpr = unaryExprOrTypeTraitExpr(has(expr(hasType(qualType(
      hasCanonicalType(anyOf(arrayType(hasElementType(isAnyCharacter())),
                             pointerType(pointee(isAnyCharacter())))))))));

  auto WrongLength =
      anyOf(ignoringImpCasts(Strlen), ignoringImpCasts(SizeOrLength),
            hasDescendant(Strlen), hasDescendant(SizeOrLength));

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
      anyOf(AnyOfCallOrDREWithoutInc, CallExprReturnWithoutInc,
            DREHasReturnWithoutInc);

  enum class StrlenKind { WithInc, WithoutInc };

  const auto AnyOfLengthExpr = [=](StrlenKind LengthKind) {
    return ignoringImpCasts(allOf(
        unless(hasDefinition(SizeOfCharExpr)),
        anyOf(allOf((LengthKind == StrlenKind::WithoutInc)
                        ? ignoringImpCasts(unless(hasDefinition(HasIncOp)))
                        : ignoringImpCasts(
                              allOf(hasDefinition(HasIncOp),
                                    unless(hasDefinition(HasDecOp)))),
                    AnyOfWrongLengthInit),
              ignoringImpCasts(integerLiteral().bind(WrongLengthExprName))),
        expr().bind(LengthExprName)));
  };

  auto LengthWithoutInc = AnyOfLengthExpr(StrlenKind::WithoutInc);
  auto LengthWithInc = AnyOfLengthExpr(StrlenKind::WithInc);

  //===--------------------------------------------------------------------===//
  // The following five cases match the 'destination' array length's
  // expression which is used in memcpy() and memmove() matchers.
  //===--------------------------------------------------------------------===//

  auto SizeExpr = anyOf(SizeOfCharExpr, integerLiteral(equals(1)));

  auto MallocLengthExpr = allOf(
      anyOf(argumentCountIs(1), argumentCountIs(2)),
      hasAnyArgument(allOf(unless(SizeExpr),
                           expr(ignoringImpCasts(anyOf(HasIncOp, anything())))
                               .bind(DestMallocExprName))));

  // - Example:  (char *)malloc(length);
  auto DestMalloc = anyOf(castExpr(has(callExpr(MallocLengthExpr))),
                          callExpr(MallocLengthExpr));

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
                        expr().bind(UnknownDestName)));

  auto AnyOfDestDecl =
      allOf(anyOf(hasDefinition(anyOf(AnyOfDestInit, DestArrayTyDecl)),
                  DestUnknownDecl, anything()),
            expr().bind(DestExprName));

  auto SrcDecl = declRefExpr(
      allOf(to(decl().bind(SrcVarDeclName)),
            anyOf(hasAncestor(cxxMemberCallExpr().bind(SrcExprName)),
                  expr().bind(SrcExprName))));

  auto SrcDeclMayInBinOp =
      anyOf(ignoringImpCasts(SrcDecl), hasDescendant(SrcDecl));

  auto AnyOfSrcDecl = anyOf(ignoringImpCasts(stringLiteral().bind(SrcExprName)),
                            SrcDeclMayInBinOp);

  auto NullTerminatorExpr = binaryOperator(
      hasLHS(hasDescendant(
          declRefExpr(to(varDecl(equalsBoundNode(DestVarDeclName)))))),
      hasRHS(ignoringImpCasts(
          anyOf(characterLiteral(equals(0U)), integerLiteral(equals(0))))));

  //===--------------------------------------------------------------------===//
  // The following nineteen cases match problematic function calls.
  //===--------------------------------------------------------------------===//

  const auto WithoutSrc = [=](StringRef Name, int LengthPos,
                              StrlenKind LengthKind) {
    return allOf(
        callee(functionDecl(hasName(Name))),
        hasArgument(
            0, allOf(AnyOfDestDecl, unless(hasAncestor(compoundStmt(
                                        hasDescendant(NullTerminatorExpr)))))),
        hasArgument(LengthPos, (LengthKind == StrlenKind::WithoutInc)
                                   ? LengthWithoutInc
                                   : LengthWithInc));
  };

  const auto WithSrc = [=](StringRef Name, int SourcePos, int LengthPos,
                           StrlenKind LengthKind) {
    return allOf(callee(functionDecl(hasName(Name))),
                 hasArgument(SourcePos ? 0 : 1,
                             allOf(AnyOfDestDecl,
                                   unless(hasAncestor(compoundStmt(
                                       hasDescendant(NullTerminatorExpr)))))),
                 hasArgument(SourcePos, AnyOfSrcDecl),
                 hasArgument(LengthPos, (LengthKind == StrlenKind::WithoutInc)
                                            ? LengthWithoutInc
                                            : LengthWithInc));
  };

  auto Memcpy = WithSrc("::memcpy", 1, 2, StrlenKind::WithoutInc);
  auto Wmemcpy = WithSrc("::wmemcpy", 1, 2, StrlenKind::WithoutInc);
  auto Memcpy_s = WithSrc("::memcpy_s", 2, 3, StrlenKind::WithoutInc);
  auto Wmemcpy_s = WithSrc("::wmemcpy_s", 2, 3, StrlenKind::WithoutInc);
  auto Memchr = WithSrc("::memchr", 0, 2, StrlenKind::WithoutInc);
  auto Wmemchr = WithSrc("::wmemchr", 0, 2, StrlenKind::WithoutInc);
  auto Memmove = WithSrc("::memmove", 1, 2, StrlenKind::WithoutInc);
  auto Wmemmove = WithSrc("::wmemmove", 1, 2, StrlenKind::WithoutInc);
  auto Memmove_s = WithSrc("::memmove_s", 2, 3, StrlenKind::WithoutInc);
  auto Wmemmove_s = WithSrc("::wmemmove_s", 2, 3, StrlenKind::WithoutInc);
  auto Memset = WithoutSrc("::memset", 2, StrlenKind::WithInc);
  auto Wmemset = WithoutSrc("::wmemset", 2, StrlenKind::WithInc);
  auto Strerror_s = WithoutSrc("::strerror_s", 1, StrlenKind::WithoutInc);
  auto StrncmpLHS = WithSrc("::strncmp", 1, 2, StrlenKind::WithInc);
  auto WcsncmpLHS = WithSrc("::wcsncmp", 1, 2, StrlenKind::WithInc);
  auto StrncmpRHS = WithSrc("::strncmp", 0, 2, StrlenKind::WithInc);
  auto WcsncmpRHS = WithSrc("::wcsncmp", 0, 2, StrlenKind::WithInc);
  auto Strxfrm = WithSrc("::strxfrm", 1, 2, StrlenKind::WithoutInc);
  auto Wcsxfrm = WithSrc("::wcsxfrm", 1, 2, StrlenKind::WithoutInc);

  auto AnyOfMatchers =
      anyOf(Memcpy, Wmemcpy, Memcpy_s, Wmemcpy_s, Memchr, Wmemchr, Memmove,
            Wmemmove, Memmove_s, Wmemmove_s, Memset, Wmemset, Strerror_s,
            StrncmpLHS, WcsncmpLHS, StrncmpRHS, WcsncmpRHS, Strxfrm, Wcsxfrm);

  Finder->addMatcher(callExpr(AnyOfMatchers).bind(FuncExprName), this);

  Finder->addMatcher(
      castExpr(has(callExpr(anyOf(Memchr, Wmemchr)).bind(FuncExprName)))
          .bind(CastExprName),
      this);
}

void NotNullTerminatedResultCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *FuncExpr = Result.Nodes.getNodeAs<CallExpr>(FuncExprName);
  if (FuncExpr->getBeginLoc().isMacroID())
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

  StringRef Name = FuncExpr->getDirectCallee()->getName();
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
      isDestAndSrcEquals(Result))
    return;

  auto Diag =
      diag(Result.Nodes.getNodeAs<CallExpr>(FuncExprName)->getBeginLoc(),
           "the result from calling '%0' is not null-terminated")
      << Name;

  if (Name.endswith("cpy"))
    memcpyFix(Name, Result, Diag);
  else if (Name.endswith("cpy_s"))
    memcpy_sFix(Name, Result, Diag);
  else if (Name.endswith("move"))
    memmoveFix(Name, Result, Diag);
  else if (Name.endswith("move_s")) {
    destCapacityFix(Result, Diag);
    lengthArgHandle(LengthHandleKind::Increase, 3, Result, Diag);
  } else if (Name.endswith("set")) {
    lengthArgHandle(LengthHandleKind::Decrease, 2, Result, Diag);
  }
}

void NotNullTerminatedResultCheck::memcpyFix(
    StringRef Name, const MatchFinder::MatchResult &Result,
    DiagnosticBuilder &Diag) {
  bool IsOverflows = destCapacityFix(Result, Diag);

  // If it cannot be rewritten to string handler function.
  if (Result.Nodes.getNodeAs<Type>(NotJustCharTyName)) {
    if (UseSafeFunctions && isKnownDest(Result)) {
      renameFunc((Name[0] != 'w') ? "memcpy_s" : "wmemcpy_s", Result, Diag);
      insertDestCapacityArg(IsOverflows, Name, Result, Diag);
    }

    lengthArgHandle(LengthHandleKind::Increase, 2, Result, Diag);
    return;
  }

  bool IsCopy =
      isLengthEqualToSrcLength(Result) || isDestBasedOnGivenLength(Result);

  bool IsSafe = UseSafeFunctions && IsOverflows && isKnownDest(Result) &&
                !isDestBasedOnGivenLength(Result);

  bool IsDestLengthNotRequired =
      IsSafe && getLangOpts().CPlusPlus &&
      Result.Nodes.getNodeAs<ArrayType>(DestArrayTyName);

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
  bool IsOverflows = destCapacityFix(Result, Diag);

  if (Result.Nodes.getNodeAs<Type>(NotJustCharTyName)) {
    lengthArgHandle(LengthHandleKind::Increase, 3, Result, Diag);
    return;
  }

  bool RemoveDestLength = getLangOpts().CPlusPlus &&
                          Result.Nodes.getNodeAs<ArrayType>(DestArrayTyName);
  bool IsCopy = isLengthEqualToSrcLength(Result);
  bool IsSafe = IsOverflows;

  renameMemcpy(Name, IsCopy, IsSafe, Result, Diag);

  if (!IsSafe || (IsSafe && RemoveDestLength))
    removeArg(1, Result, Diag);
  else if (IsOverflows && isKnownDest(Result))
    lengthArgHandle(LengthHandleKind::Increase, 1, Result, Diag);

  if (IsCopy)
    removeArg(3, Result, Diag);

  if (!IsCopy && !IsSafe)
    insertNullTerminatorExpr(Name, Result, Diag);
}

void NotNullTerminatedResultCheck::memchrFix(
    StringRef Name, const MatchFinder::MatchResult &Result) {
  const auto *FuncExpr = Result.Nodes.getNodeAs<CallExpr>(FuncExprName);
  if (const auto GivenCL =
          dyn_cast_or_null<CharacterLiteral>(FuncExpr->getArg(1)))
    if (GivenCL->getValue() != 0)
      return;

  auto Diag = diag(FuncExpr->getArg(2)->IgnoreParenCasts()->getBeginLoc(),
                   "the length is too short to include the null terminator");

  if (const auto *CastExpr = Result.Nodes.getNodeAs<Expr>(CastExprName)) {
    const auto CastRemoveFix = FixItHint::CreateRemoval(SourceRange(
        CastExpr->getBeginLoc(), FuncExpr->getBeginLoc().getLocWithOffset(-1)));
    Diag << CastRemoveFix;
  }
  StringRef NewFuncName = (Name[0] != 'w') ? "strchr" : "wcschr";
  renameFunc(NewFuncName, Result, Diag);
  removeArg(2, Result, Diag);
}

void NotNullTerminatedResultCheck::memmoveFix(
    StringRef Name, const MatchFinder::MatchResult &Result,
    DiagnosticBuilder &Diag) {
  bool IsOverflows = destCapacityFix(Result, Diag);

  if (UseSafeFunctions && isKnownDest(Result)) {
    renameFunc((Name[0] != 'w') ? "memmove_s" : "wmemmove_s", Result, Diag);
    insertDestCapacityArg(IsOverflows, Name, Result, Diag);
  }

  lengthArgHandle(LengthHandleKind::Increase, 2, Result, Diag);
}

void NotNullTerminatedResultCheck::strerror_sFix(
    const MatchFinder::MatchResult &Result) {
  StringRef Name = "strerror_s";
  auto Diag =
      diag(Result.Nodes.getNodeAs<CallExpr>(FuncExprName)->getBeginLoc(),
           "the result from calling '%0' is not null-terminated and "
           "missing the last character of the error message")
      << Name;

  destCapacityFix(Result, Diag);
  lengthArgHandle(LengthHandleKind::Increase, 1, Result, Diag);
}

void NotNullTerminatedResultCheck::ncmpFix(
    StringRef Name, const MatchFinder::MatchResult &Result) {
  const auto *FuncExpr = Result.Nodes.getNodeAs<CallExpr>(FuncExprName);
  const Expr *FirstArgExpr = FuncExpr->getArg(0)->IgnoreImpCasts();
  const Expr *SecondArgExpr = FuncExpr->getArg(1)->IgnoreImpCasts();
  bool IsLengthTooLong = false;

  if (const auto *LengthExpr =
          Result.Nodes.getNodeAs<CallExpr>(WrongLengthExprName)) {
    const Expr *LengthExprArg = LengthExpr->getArg(0);
    StringRef FirstExprStr = exprToStr(FirstArgExpr, Result).trim(' ');
    StringRef SecondExprStr = exprToStr(SecondArgExpr, Result).trim(' ');
    StringRef LengthArgStr = exprToStr(LengthExprArg, Result).trim(' ');
    IsLengthTooLong =
        LengthArgStr == FirstExprStr || LengthArgStr == SecondExprStr;
  } else {
    int SrcLength =
        getLength(Result.Nodes.getNodeAs<Expr>(SrcExprName), Result);
    int GivenLength = getGivenLength(Result);
    IsLengthTooLong = GivenLength - 1 == SrcLength;
  }

  if (!IsLengthTooLong && !isStringDataAndLength(Result))
    return;

  auto Diag = diag(FuncExpr->getArg(2)->IgnoreParenCasts()->getBeginLoc(),
                   "comparison length is too long and might lead to a "
                   "buffer overflow");

  lengthArgHandle(LengthHandleKind::Decrease, 2, Result, Diag);
}

void NotNullTerminatedResultCheck::xfrmFix(
    StringRef Name, const MatchFinder::MatchResult &Result) {
  if (!isDestCapacityOverflows(Result))
    return;

  auto Diag =
      diag(Result.Nodes.getNodeAs<CallExpr>(FuncExprName)->getBeginLoc(),
           "the result from calling '%0' is not null-terminated")
      << Name;

  destCapacityFix(Result, Diag);
  lengthArgHandle(LengthHandleKind::Increase, 2, Result, Diag);
}

//===---------------------------------------------------------------------===//
// All the helper functions.
//===---------------------------------------------------------------------===//

static StringRef exprToStr(const Expr *E,
                           const MatchFinder::MatchResult &Result) {
  if (!E)
    return "";

  return Lexer::getSourceText(
      CharSourceRange::getTokenRange(E->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts(), 0);
}

static bool isDestAndSrcEquals(const MatchFinder::MatchResult &Result) {
  if (const auto *DestVD = Result.Nodes.getNodeAs<Decl>(DestVarDeclName))
    if (const auto *SrcVD = Result.Nodes.getNodeAs<Decl>(SrcVarDeclName))
      return DestVD->getCanonicalDecl() == SrcVD->getCanonicalDecl();

  return false;
}

static bool isCorrectGivenLength(const MatchFinder::MatchResult &Result) {
  if (Result.Nodes.getNodeAs<IntegerLiteral>(WrongLengthExprName))
    return !isLengthEqualToSrcLength(Result);

  return false;
}

static const Expr *getDestCapacityExpr(const MatchFinder::MatchResult &Result) {
  if (const auto *DestMalloc = Result.Nodes.getNodeAs<Expr>(DestMallocExprName))
    return DestMalloc;

  if (const auto *DestTy = Result.Nodes.getNodeAs<ArrayType>(DestArrayTyName))
    if (const auto *DestVAT = dyn_cast_or_null<VariableArrayType>(DestTy))
      return DestVAT->getSizeExpr();

  if (const auto *DestVD = Result.Nodes.getNodeAs<VarDecl>(DestVarDeclName))
    if (const auto DestTL = DestVD->getTypeSourceInfo()->getTypeLoc())
      if (const auto DestCTL = DestTL.getAs<ConstantArrayTypeLoc>())
        return DestCTL.getSizeExpr();

  return nullptr;
}

static int getLength(const Expr *E, const MatchFinder::MatchResult &Result) {
  llvm::APSInt Length;

  if (const auto *LengthDRE = dyn_cast_or_null<DeclRefExpr>(E))
    if (const auto *LengthVD = dyn_cast_or_null<VarDecl>(LengthDRE->getDecl()))
      if (!isa<ParmVarDecl>(LengthVD))
        if (const Expr *LengthInit = LengthVD->getInit())
          if (LengthInit->EvaluateAsInt(Length, *Result.Context))
            return Length.getZExtValue();

  if (const auto *LengthIL = dyn_cast_or_null<IntegerLiteral>(E))
    return LengthIL->getValue().getZExtValue();

  if (const auto *StrDRE = dyn_cast_or_null<DeclRefExpr>(E))
    if (const auto *StrVD = dyn_cast_or_null<VarDecl>(StrDRE->getDecl()))
      if (const Expr *StrInit = StrVD->getInit())
        if (const auto *StrSL =
                dyn_cast_or_null<StringLiteral>(StrInit->IgnoreImpCasts()))
          return StrSL->getLength();

  if (const auto *SrcSL = dyn_cast_or_null<StringLiteral>(E))
    return SrcSL->getLength();

  return 0;
}

static int getDestCapacity(const MatchFinder::MatchResult &Result) {
  if (const auto *DestCapacityExpr = getDestCapacityExpr(Result))
    return getLength(DestCapacityExpr, Result);

  return 0;
}

static int getGivenLength(const MatchFinder::MatchResult &Result) {
  const auto *LengthExpr = Result.Nodes.getNodeAs<Expr>(LengthExprName);
  if (int Length = getLength(LengthExpr, Result))
    return Length;

  if (const auto *StrlenExpr = dyn_cast_or_null<CallExpr>(LengthExpr))
    if (StrlenExpr->getNumArgs() > 0)
      if (const Expr *StrlenArg = StrlenExpr->getArg(0)->IgnoreImpCasts())
        if (int StrlenArgLength = getLength(StrlenArg, Result))
          return StrlenArgLength;

  return 0;
}

static bool isStringDataAndLength(const MatchFinder::MatchResult &Result) {
  StringRef DestStr =
      exprToStr(Result.Nodes.getNodeAs<Expr>(DestExprName), Result);
  StringRef SrcStr =
      exprToStr(Result.Nodes.getNodeAs<Expr>(SrcExprName), Result);
  StringRef GivenLengthStr =
      exprToStr(Result.Nodes.getNodeAs<Expr>(LengthExprName), Result);

  bool ProblematicLength =
      GivenLengthStr.contains(".size") || GivenLengthStr.contains(".length");

  return ProblematicLength &&
         (SrcStr.contains(".data") || DestStr.contains(".data"));
}

static bool isLengthEqualToSrcLength(const MatchFinder::MatchResult &Result) {
  if (isStringDataAndLength(Result))
    return true;

  int GivenLength = getGivenLength(Result);

  // It is the length without the null terminator.
  int SrcLength = getLength(Result.Nodes.getNodeAs<Expr>(SrcExprName), Result);

  if (GivenLength != 0 && GivenLength == SrcLength)
    return true;

  // If 'strlen()' check the VarDecl of the argument is equal to source VarDecl.
  if (const auto *StrlenExpr = Result.Nodes.getNodeAs<CallExpr>(LengthExprName))
    if (StrlenExpr->getNumArgs() > 0)
      if (const auto *StrlenDRE = dyn_cast_or_null<DeclRefExpr>(
              StrlenExpr->getArg(0)->IgnoreImpCasts()))
        if (const auto *SrcVD = Result.Nodes.getNodeAs<VarDecl>(SrcVarDeclName))
          return dyn_cast_or_null<VarDecl>(StrlenDRE->getDecl()) == SrcVD;

  return false;
}

static bool isDestCapacityOverflows(const MatchFinder::MatchResult &Result) {
  if (!isKnownDest(Result))
    return true;

  const auto *DestCapacityExpr = getDestCapacityExpr(Result);
  const auto *LengthExpr = Result.Nodes.getNodeAs<Expr>(LengthExprName);
  int DestCapacity = getLength(DestCapacityExpr, Result);
  int GivenLength = getGivenLength(Result);

  if (GivenLength != 0 && DestCapacity != 0)
    return isLengthEqualToSrcLength(Result) && DestCapacity == GivenLength;

  StringRef DestCapacityExprStr = exprToStr(DestCapacityExpr, Result);
  StringRef LengthExprStr = exprToStr(LengthExpr, Result);

  // Assume that it cannot overflow if the expression of the destination
  // capacity contains '+ 1'.
  if (DestCapacityExprStr.contains("+1") || DestCapacityExprStr.contains("+ 1"))
    return false;

  if (DestCapacityExprStr != "" && DestCapacityExprStr == LengthExprStr)
    return true;

  return true;
}

static bool isDestBasedOnGivenLength(const MatchFinder::MatchResult &Result) {
  StringRef DestCapacityExprStr =
      exprToStr(getDestCapacityExpr(Result), Result).trim(' ');
  StringRef LengthExprStr =
      exprToStr(Result.Nodes.getNodeAs<Expr>(LengthExprName), Result).trim(' ');

  return DestCapacityExprStr != "" && LengthExprStr != "" &&
         DestCapacityExprStr.contains(LengthExprStr);
}

static void lengthDecrease(const Expr *LengthExpr,
                           const MatchFinder::MatchResult &Result,
                           DiagnosticBuilder &Diag) {
  // This is the following structure: ((strlen(src) * 2) + 1)
  //                     InnerOpExpr:   ~~~~~~~~~~~~^~~
  //                     OuterOpExpr:  ~~~~~~~~~~~~~~~~~~^~~
  if (const auto *OuterOpExpr =
          dyn_cast_or_null<BinaryOperator>(LengthExpr->IgnoreParenCasts())) {
    const Expr *LHSExpr = OuterOpExpr->getLHS();
    const Expr *RHSExpr = OuterOpExpr->getRHS();
    const auto *InnerOpExpr =
        isa<IntegerLiteral>(RHSExpr->IgnoreCasts()) ? LHSExpr : RHSExpr;

    // This is the following structure: ((strlen(src) * 2) + 1)
    //                  LHSRemoveRange: ~~
    //                  RHSRemoveRange:                  ~~~~~~
    SourceRange LHSRemoveRange(LengthExpr->getBeginLoc(),
                               InnerOpExpr->getBeginLoc().getLocWithOffset(-1));
    SourceRange RHSRemoveRange(exprLocEnd(InnerOpExpr, Result),
                               LengthExpr->getEndLoc());
    const auto LHSRemoveFix = FixItHint::CreateRemoval(LHSRemoveRange);
    const auto RHSRemoveFix = FixItHint::CreateRemoval(RHSRemoveRange);

    if (LengthExpr->getBeginLoc() == InnerOpExpr->getBeginLoc())
      Diag << RHSRemoveFix;
    else if (LengthExpr->getEndLoc() == InnerOpExpr->getEndLoc())
      Diag << LHSRemoveFix;
    else
      Diag << LHSRemoveFix << RHSRemoveFix;
  } else {
    const auto InsertDecreaseFix =
        FixItHint::CreateInsertion(exprLocEnd(LengthExpr, Result), " - 1");
    Diag << InsertDecreaseFix;
  }
}

static void lengthIncrease(const Expr *LengthExpr,
                           const MatchFinder::MatchResult &Result,
                           DiagnosticBuilder &Diag) {
  bool NeedInnerParen = dyn_cast_or_null<BinaryOperator>(LengthExpr) &&
                        cast<BinaryOperator>(LengthExpr)->getOpcode() != BO_Add;

  if (NeedInnerParen) {
    const auto InsertFirstParenFix =
        FixItHint::CreateInsertion(LengthExpr->getBeginLoc(), "(");
    const auto InsertPlusOneAndSecondParenFix =
        FixItHint::CreateInsertion(exprLocEnd(LengthExpr, Result),
                                   !isInjectUL(Result) ? ") + 1" : ") + 1UL");
    Diag << InsertFirstParenFix << InsertPlusOneAndSecondParenFix;
  } else {
    const auto InsertPlusOneFix =
        FixItHint::CreateInsertion(exprLocEnd(LengthExpr, Result),
                                   !isInjectUL(Result) ? " + 1" : " + 1UL");
    Diag << InsertPlusOneFix;
  }
}

static void lengthExprHandle(LengthHandleKind LengthHandle,
                             const Expr *LengthExpr,
                             const MatchFinder::MatchResult &Result,
                             DiagnosticBuilder &Diag) {
  if (!LengthExpr)
    return;

  bool IsMacroDefinition = false;
  StringRef LengthExprStr = exprToStr(LengthExpr, Result);

  Preprocessor::macro_iterator It = PP->macro_begin();
  while (It != PP->macro_end() && !IsMacroDefinition) {
    if (It->first->getName() == LengthExprStr)
      IsMacroDefinition = true;

    ++It;
  }

  if (!IsMacroDefinition) {
    if (const auto *LengthIL = dyn_cast_or_null<IntegerLiteral>(LengthExpr)) {
      const size_t NewLength = LengthIL->getValue().getZExtValue() +
                               (LengthHandle == LengthHandleKind::Increase
                                    ? (isInjectUL(Result) ? 1UL : 1)
                                    : -1);
      const auto NewLengthFix = FixItHint::CreateReplacement(
          LengthIL->getSourceRange(),
          (Twine(NewLength) + (isInjectUL(Result) ? "UL" : "")).str());
      Diag << NewLengthFix;
      return;
    }

    if (LengthHandle == LengthHandleKind::Increase)
      lengthIncrease(LengthExpr, Result, Diag);
    else
      lengthDecrease(LengthExpr, Result, Diag);
  } else {
    if (LengthHandle == LengthHandleKind::Increase) {
      const auto InsertPlusOneFix =
          FixItHint::CreateInsertion(exprLocEnd(LengthExpr, Result),
                                     !isInjectUL(Result) ? " + 1" : " + 1UL");
      Diag << InsertPlusOneFix;
    } else {
      const auto InsertMinusOneFix =
          FixItHint::CreateInsertion(exprLocEnd(LengthExpr, Result), " - 1");
      Diag << InsertMinusOneFix;
    }
  }
}

static void lengthArgHandle(LengthHandleKind LengthHandle, int ArgPos,
                            const MatchFinder::MatchResult &Result,
                            DiagnosticBuilder &Diag) {
  const auto *FuncExpr = Result.Nodes.getNodeAs<CallExpr>(FuncExprName);
  const Expr *LengthExpr = FuncExpr->getArg(ArgPos)->IgnoreImpCasts();
  lengthExprHandle(LengthHandle, LengthExpr, Result, Diag);
}

static bool destCapacityFix(const MatchFinder::MatchResult &Result,
                            DiagnosticBuilder &Diag) {
  bool IsOverflows = isDestCapacityOverflows(Result);
  if (IsOverflows)
    lengthExprHandle(LengthHandleKind::Increase, getDestCapacityExpr(Result),
                     Result, Diag);

  return IsOverflows;
}

static void removeArg(int ArgPos, const MatchFinder::MatchResult &Result,
                      DiagnosticBuilder &Diag) {
  // This is the following structure: (src, '\0', strlen(src))
  //                     ArgToRemove:             ~~~~~~~~~~~
  //                          LHSArg:       ~~~~
  //                    RemoveArgFix:           ~~~~~~~~~~~~~
  const auto *FuncExpr = Result.Nodes.getNodeAs<CallExpr>(FuncExprName);
  const Expr *ArgToRemove = FuncExpr->getArg(ArgPos);
  const Expr *LHSArg = FuncExpr->getArg(ArgPos - 1);
  const auto RemoveArgFix = FixItHint::CreateRemoval(
      SourceRange(exprLocEnd(LHSArg, Result),
                  exprLocEnd(ArgToRemove, Result).getLocWithOffset(-1)));
  Diag << RemoveArgFix;
}

static void renameFunc(StringRef NewFuncName,
                       const MatchFinder::MatchResult &Result,
                       DiagnosticBuilder &Diag) {
  const auto *FuncExpr = Result.Nodes.getNodeAs<CallExpr>(FuncExprName);
  int FuncNameLength =
      FuncExpr->getDirectCallee()->getIdentifier()->getLength();
  SourceRange FuncNameRange(
      FuncExpr->getBeginLoc(),
      FuncExpr->getBeginLoc().getLocWithOffset(FuncNameLength - 1));

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
  const auto *FuncExpr = Result.Nodes.getNodeAs<CallExpr>(FuncExprName);
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
      FuncExpr->getArg(1)->getBeginLoc(), NewSecondArg);
  Diag << InsertNewArgFix;
}

static void insertNullTerminatorExpr(StringRef Name,
                                     const MatchFinder::MatchResult &Result,
                                     DiagnosticBuilder &Diag) {
  const auto *FuncExpr = Result.Nodes.getNodeAs<CallExpr>(FuncExprName);
  int FuncLocStartColumn =
      Result.SourceManager->getPresumedColumnNumber(FuncExpr->getBeginLoc());
  SourceRange SpaceRange(
      FuncExpr->getBeginLoc().getLocWithOffset(-FuncLocStartColumn + 1),
      FuncExpr->getBeginLoc());
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
      exprLocEnd(FuncExpr, Result).getLocWithOffset(1), NewAddNullTermExprStr);
  Diag << AddNullTerminatorExprFix;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
