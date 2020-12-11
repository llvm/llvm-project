//===- OSObjectCStyleCast.cpp ------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines OSObjectCStyleCast checker, which checks for C-style casts
// of OSObjects. Such casts almost always indicate a code smell,
// as an explicit static or dynamic cast should be used instead.
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "llvm/Support/Debug.h"

using namespace clang;
using namespace ento;
using namespace ast_matchers;

namespace {
static constexpr const char *const WarnAtNode = "WarnAtNode";
static constexpr const char *const WarnRecordDecl = "WarnRecordDecl";

class OSObjectCStyleCastChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager &AM,
                        BugReporter &BR) const;
};
}

static void emitDiagnostics(const BoundNodes &Nodes,
                            BugReporter &BR,
                            AnalysisDeclContext *ADC,
                            const OSObjectCStyleCastChecker *Checker) {
  const auto *CE = Nodes.getNodeAs<CastExpr>(WarnAtNode);
  const CXXRecordDecl *RD = Nodes.getNodeAs<CXXRecordDecl>(WarnRecordDecl);
  assert(CE && RD);

  std::string Diagnostics;
  llvm::raw_string_ostream OS(Diagnostics);
  OS << "C-style cast of an OSObject is prone to type confusion attacks; "
     << "use 'OSRequiredCast' if the object is definitely of type '"
     << RD->getNameAsString() << "', or 'OSDynamicCast' followed by "
     << "a null check if unsure",

  BR.EmitBasicReport(
    ADC->getDecl(),
    Checker,
    /*Name=*/"OSObject C-Style Cast",
    categories::SecurityError,
    OS.str(),
    PathDiagnosticLocation::createBegin(CE, BR.getSourceManager(), ADC),
    CE->getSourceRange());
}

static decltype(auto) hasTypePointingTo(DeclarationMatcher DeclM) {
  return hasType(pointerType(pointee(hasDeclaration(DeclM))));
}

void OSObjectCStyleCastChecker::checkASTCodeBody(const Decl *D, AnalysisManager &AM,
                                                 BugReporter &BR) const {

  AnalysisDeclContext *ADC = AM.getAnalysisDeclContext(D);

  auto DynamicCastM = callExpr(callee(functionDecl(hasName("safeMetaCast"))));

  auto OSObjTypeM = hasTypePointingTo(cxxRecordDecl(isDerivedFrom("OSMetaClassBase")));
  auto OSObjSubclassM = hasTypePointingTo(
    cxxRecordDecl(isDerivedFrom("OSObject")).bind(WarnRecordDecl));

  auto CastM = cStyleCastExpr(
          allOf(hasSourceExpression(allOf(OSObjTypeM, unless(DynamicCastM))),
          OSObjSubclassM)).bind(WarnAtNode);

  auto Matches = match(stmt(forEachDescendant(CastM)), *D->getBody(), AM.getASTContext());
  for (BoundNodes Match : Matches)
    emitDiagnostics(Match, BR, ADC, this);
}

void ento::registerOSObjectCStyleCast(CheckerManager &Mgr) {
  Mgr.registerChecker<OSObjectCStyleCastChecker>();
}

bool ento::shouldRegisterOSObjectCStyleCast(const CheckerManager &mgr) {
  return true;
}
