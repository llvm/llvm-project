//===- NumberObjectConversionChecker.cpp -------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines NumberObjectConversionChecker, which checks for a
// particular common mistake when dealing with numbers represented as objects
// passed around by pointers. Namely, the language allows to reinterpret the
// pointer as a number directly, often without throwing any warnings,
// but in most cases the result of such conversion is clearly unexpected,
// as pointer value, rather than number value represented by the pointee object,
// becomes the result of such operation.
//
// Currently the checker supports the Objective-C NSNumber class,
// and the OSBoolean class found in macOS low-level code; the latter
// can only hold boolean values.
//
// This checker has an option "Pedantic" (boolean), which enables detection of
// more conversion patterns (which are most likely more harmless, and therefore
// are more likely to produce false positives) - disabled by default,
// enabled with `-analyzer-config osx.NumberObjectConversion:Pedantic=true'.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/APSInt.h"

using namespace clang;
using namespace ento;
using namespace ast_matchers;

namespace {

class NumberObjectConversionChecker : public Checker<check::ASTCodeBody> {
public:
  bool Pedantic;

  void checkASTCodeBody(const Decl *D, AnalysisManager &AM,
                        BugReporter &BR) const;
};

class Callback : public MatchFinder::MatchCallback {
  const NumberObjectConversionChecker *C;
  BugReporter &BR;
  AnalysisDeclContext *ADC;

public:
  Callback(const NumberObjectConversionChecker *C,
           BugReporter &BR, AnalysisDeclContext *ADC)
      : C(C), BR(BR), ADC(ADC) {}
  virtual void run(const MatchFinder::MatchResult &Result);
};
} // end of anonymous namespace

void Callback::run(const MatchFinder::MatchResult &Result) {
  bool IsPedanticMatch = (Result.Nodes.getNodeAs<Stmt>("pedantic") != nullptr);
  if (IsPedanticMatch && !C->Pedantic)
    return;

  const Stmt *Conv = Result.Nodes.getNodeAs<Stmt>("conv");
  assert(Conv);
  const Expr *Osboolean = Result.Nodes.getNodeAs<Expr>("osboolean");
  const Expr *Nsnumber = Result.Nodes.getNodeAs<Expr>("nsnumber");
  bool IsObjC = (bool)Nsnumber;
  const Expr *Obj = IsObjC ? Nsnumber : Osboolean;
  assert(Obj);

  ASTContext &ACtx = ADC->getASTContext();

  if (const Expr *CheckIfNull =
          Result.Nodes.getNodeAs<Expr>("check_if_null")) {
    // We consider NULL to be a pointer, even if it is defined as a plain 0.
    // FIXME: Introduce a matcher to implement this logic?
    SourceLocation Loc = CheckIfNull->getLocStart();
    if (Loc.isMacroID()) {
      StringRef MacroName = Lexer::getImmediateMacroName(
          Loc, ACtx.getSourceManager(), ACtx.getLangOpts());
      if (MacroName != "YES" && MacroName != "NO")
        return;
    } else {
      // Otherwise, comparison of pointers to 0 might still be intentional.
      // See if this is the case.
      llvm::APSInt Result;
      if (CheckIfNull->IgnoreParenCasts()->EvaluateAsInt(
              Result, ACtx, Expr::SE_AllowSideEffects)) {
        if (Result == 0) {
          if (!C->Pedantic)
            return;
          IsPedanticMatch = true;
        }
      }
    }
  }

  llvm::SmallString<64> Msg;
  llvm::raw_svector_ostream OS(Msg);
  OS << "Converting '"
     << Obj->getType().getCanonicalType().getUnqualifiedType().getAsString()
     << "' to a plain ";

  if (Result.Nodes.getNodeAs<QualType>("int_type") != nullptr)
    OS << "integer value";
  else if (Result.Nodes.getNodeAs<QualType>("objc_bool_type") != nullptr)
    OS << "BOOL value";
  else if (Result.Nodes.getNodeAs<QualType>("cpp_bool_type") != nullptr)
    OS << "bool value";
  else
    OS << "boolean value for branching";

  if (IsPedanticMatch) {
    if (IsObjC) {
      OS << "; please compare the pointer to nil instead "
            "to suppress this warning";
    } else {
      OS << "; please compare the pointer to NULL or nullptr instead "
            "to suppress this warning";
    }
  } else {
    OS << "; pointer value is being used instead";
  }

  BR.EmitBasicReport(
      ADC->getDecl(), C, "Suspicious number object conversion", "Logic error",
      OS.str(),
      PathDiagnosticLocation::createBegin(Obj, BR.getSourceManager(), ADC),
      Conv->getSourceRange());
}

void NumberObjectConversionChecker::checkASTCodeBody(const Decl *D,
                                                     AnalysisManager &AM,
                                                     BugReporter &BR) const {
  MatchFinder F;
  Callback CB(this, BR, AM.getAnalysisDeclContext(D));

  auto OSBooleanExprM =
      expr(ignoringParenImpCasts(
          expr(hasType(hasCanonicalType(
              pointerType(pointee(hasCanonicalType(
                  recordType(hasDeclaration(
                      cxxRecordDecl(hasName(
                          "OSBoolean")))))))))).bind("osboolean")));

  auto NSNumberExprM =
      expr(ignoringParenImpCasts(expr(hasType(hasCanonicalType(
          objcObjectPointerType(pointee(
              qualType(hasCanonicalType(
                  qualType(hasDeclaration(
                      objcInterfaceDecl(hasName(
                          "NSNumber"))))))))))).bind("nsnumber")));

  auto SuspiciousExprM =
      anyOf(OSBooleanExprM, NSNumberExprM);

  auto AnotherNSNumberExprM =
      expr(equalsBoundNode("nsnumber"));

  // The .bind here is in order to compose the error message more accurately.
  auto ObjCBooleanTypeM =
      qualType(typedefType(hasDeclaration(
                   typedefDecl(hasName("BOOL"))))).bind("objc_bool_type");

  // The .bind here is in order to compose the error message more accurately.
  auto AnyBooleanTypeM =
      qualType(anyOf(qualType(booleanType()).bind("cpp_bool_type"),
                     ObjCBooleanTypeM));


  // The .bind here is in order to compose the error message more accurately.
  auto AnyNumberTypeM =
      qualType(hasCanonicalType(isInteger()),
               unless(typedefType(hasDeclaration(
                   typedefDecl(matchesName("^::u?intptr_t$"))))))
      .bind("int_type");

  auto AnyBooleanOrNumberTypeM =
      qualType(anyOf(AnyBooleanTypeM, AnyNumberTypeM));

  auto AnyBooleanOrNumberExprM =
      expr(ignoringParenImpCasts(expr(hasType(AnyBooleanOrNumberTypeM))));

  auto ConversionThroughAssignmentM =
      binaryOperator(hasOperatorName("="),
                     hasLHS(AnyBooleanOrNumberExprM),
                     hasRHS(SuspiciousExprM));

  auto ConversionThroughBranchingM =
      ifStmt(hasCondition(SuspiciousExprM))
      .bind("pedantic");

  auto ConversionThroughCallM =
      callExpr(hasAnyArgument(allOf(hasType(AnyBooleanOrNumberTypeM),
                                    ignoringParenImpCasts(SuspiciousExprM))));

  // We bind "check_if_null" to modify the warning message
  // in case it was intended to compare a pointer to 0 with a relatively-ok
  // construct "x == 0" or "x != 0".
  auto ConversionThroughEquivalenceM =
      binaryOperator(anyOf(hasOperatorName("=="), hasOperatorName("!=")),
                     hasEitherOperand(SuspiciousExprM),
                     hasEitherOperand(AnyBooleanOrNumberExprM
                                      .bind("check_if_null")));

  auto ConversionThroughComparisonM =
      binaryOperator(anyOf(hasOperatorName(">="), hasOperatorName(">"),
                           hasOperatorName("<="), hasOperatorName("<")),
                     hasEitherOperand(SuspiciousExprM),
                     hasEitherOperand(AnyBooleanOrNumberExprM));

  auto ConversionThroughConditionalOperatorM =
      conditionalOperator(
          hasCondition(SuspiciousExprM),
          unless(hasTrueExpression(hasDescendant(AnotherNSNumberExprM))),
          unless(hasFalseExpression(hasDescendant(AnotherNSNumberExprM))))
      .bind("pedantic");

  auto ConversionThroughExclamationMarkM =
      unaryOperator(hasOperatorName("!"), has(expr(SuspiciousExprM)))
      .bind("pedantic");

  auto ConversionThroughExplicitBooleanCastM =
      explicitCastExpr(hasType(AnyBooleanTypeM),
                       has(expr(SuspiciousExprM)))
      .bind("pedantic");

  auto ConversionThroughExplicitNumberCastM =
      explicitCastExpr(hasType(AnyNumberTypeM),
                       has(expr(SuspiciousExprM)));

  auto ConversionThroughInitializerM =
      declStmt(hasSingleDecl(
          varDecl(hasType(AnyBooleanOrNumberTypeM),
                  hasInitializer(SuspiciousExprM))));

  auto FinalM = stmt(anyOf(ConversionThroughAssignmentM,
                           ConversionThroughBranchingM,
                           ConversionThroughCallM,
                           ConversionThroughComparisonM,
                           ConversionThroughConditionalOperatorM,
                           ConversionThroughEquivalenceM,
                           ConversionThroughExclamationMarkM,
                           ConversionThroughExplicitBooleanCastM,
                           ConversionThroughExplicitNumberCastM,
                           ConversionThroughInitializerM)).bind("conv");

  F.addMatcher(stmt(forEachDescendant(FinalM)), &CB);
  F.match(*D->getBody(), AM.getASTContext());
}

void ento::registerNumberObjectConversionChecker(CheckerManager &Mgr) {
  const LangOptions &LO = Mgr.getLangOpts();
  if (LO.CPlusPlus || LO.ObjC2) {
    NumberObjectConversionChecker *Chk =
        Mgr.registerChecker<NumberObjectConversionChecker>();
    Chk->Pedantic =
        Mgr.getAnalyzerOptions().getBooleanOption("Pedantic", false, Chk);
  }
}
