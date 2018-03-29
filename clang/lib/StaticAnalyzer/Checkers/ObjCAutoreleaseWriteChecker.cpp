//===- ObjCAutoreleaseWriteChecker.cpp ----------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ObjCAutoreleaseWriteChecker which warns against writes
// into autoreleased out parameters which are likely to cause crashes.
// An example of a problematic write is a write to {@code error} in the example
// below:
//
// - (BOOL) mymethod:(NSError *__autoreleasing *)error list:(NSArray*) list {
//     [list enumerateObjectsUsingBlock:^(id obj, NSUInteger idx, BOOL *stop) {
//       NSString *myString = obj;
//       if ([myString isEqualToString:@"error"] && error)
//         *error = [NSError errorWithDomain:@"MyDomain" code:-1];
//     }];
//     return false;
// }
//
// Such code is very likely to crash due to the other queue autorelease pool
// begin able to free the error object.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "llvm/ADT/Twine.h"

using namespace clang;
using namespace ento;
using namespace ast_matchers;

namespace {

const char *ProblematicWriteBind = "problematicwrite";
const char *ParamBind = "parambind";
const char *MethodBind = "methodbind";

class ObjCAutoreleaseWriteChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D,
                        AnalysisManager &AM,
                        BugReporter &BR) const;
private:
  std::vector<std::string> SelectorsWithAutoreleasingPool = {
      "enumerateObjectsUsingBlock:",
      "enumerateKeysAndObjectsUsingBlock:",
      "enumerateKeysAndObjectsWithOptions:usingBlock:",
      "enumerateObjectsWithOptions:usingBlock:",
      "enumerateObjectsAtIndexes:options:usingBlock:",
      "enumerateIndexesWithOptions:usingBlock:",
      "enumerateIndexesUsingBlock:",
      "enumerateIndexesInRange:options:usingBlock:",
      "enumerateRangesUsingBlock:",
      "enumerateRangesWithOptions:usingBlock:",
      "enumerateRangesInRange:options:usingBlock:"
      "objectWithOptions:passingTest:",
  };

  std::vector<std::string> FunctionsWithAutoreleasingPool = {
      "dispatch_async", "dispatch_group_async", "dispatch_barrier_async"};
};
}

static inline std::vector<llvm::StringRef> toRefs(std::vector<std::string> V) {
  return std::vector<llvm::StringRef>(V.begin(), V.end());
}

static auto callsNames(std::vector<std::string> FunctionNames)
    -> decltype(callee(functionDecl())) {
  return callee(functionDecl(hasAnyName(toRefs(FunctionNames))));
}

static void emitDiagnostics(BoundNodes &Match, const Decl *D, BugReporter &BR,
                            AnalysisManager &AM,
                            const ObjCAutoreleaseWriteChecker *Checker) {
  AnalysisDeclContext *ADC = AM.getAnalysisDeclContext(D);

  const auto *PVD = Match.getNodeAs<ParmVarDecl>(ParamBind);
  assert(PVD);
  QualType Ty = PVD->getType();
  if (Ty->getPointeeType().getObjCLifetime() != Qualifiers::OCL_Autoreleasing)
    return;
  const auto *SW = Match.getNodeAs<Expr>(ProblematicWriteBind);
  bool IsMethod = Match.getNodeAs<ObjCMethodDecl>(MethodBind) != nullptr;
  const char *Name = IsMethod ? "method" : "function";
  assert(SW);
  BR.EmitBasicReport(
      ADC->getDecl(), Checker,
      /*Name=*/"Writing into auto-releasing variable from a different queue",
      /*Category=*/"Memory",
      (llvm::Twine("Writing into an auto-releasing out parameter inside ") +
       "autorelease pool that may exit before " + Name + " returns; consider "
       "writing first to a strong local variable declared outside of the block")
          .str(),
      PathDiagnosticLocation::createBegin(SW, BR.getSourceManager(), ADC),
      SW->getSourceRange());
}

void ObjCAutoreleaseWriteChecker::checkASTCodeBody(const Decl *D,
                                                  AnalysisManager &AM,
                                                  BugReporter &BR) const {

  // Write into a binded object, e.g. *ParamBind = X.
  auto WritesIntoM = binaryOperator(
    hasLHS(unaryOperator(
        hasOperatorName("*"),
        hasUnaryOperand(
          ignoringParenImpCasts(
            declRefExpr(to(parmVarDecl(equalsBoundNode(ParamBind))))))
    )),
    hasOperatorName("=")
  ).bind(ProblematicWriteBind);

  // WritesIntoM happens inside a block passed as an argument.
  auto WritesInBlockM = hasAnyArgument(allOf(
      hasType(hasCanonicalType(blockPointerType())),
      forEachDescendant(WritesIntoM)
      ));

  auto CallsAsyncM = stmt(anyOf(
    callExpr(allOf(
      callsNames(FunctionsWithAutoreleasingPool), WritesInBlockM)),
    objcMessageExpr(allOf(
       hasAnySelector(toRefs(SelectorsWithAutoreleasingPool)),
       WritesInBlockM))
  ));

  auto DoublePointerParamM =
      parmVarDecl(hasType(pointerType(
                      pointee(hasCanonicalType(objcObjectPointerType())))))
          .bind(ParamBind);

  auto HasParamAndWritesAsyncM = allOf(
      hasAnyParameter(DoublePointerParamM),
      forEachDescendant(CallsAsyncM));

  auto MatcherM = decl(anyOf(
      objcMethodDecl(HasParamAndWritesAsyncM).bind(MethodBind),
      functionDecl(HasParamAndWritesAsyncM)));

  auto Matches = match(MatcherM, *D, AM.getASTContext());
  for (BoundNodes Match : Matches)
    emitDiagnostics(Match, D, BR, AM, this);
}

void ento::registerAutoreleaseWriteChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<ObjCAutoreleaseWriteChecker>();
}
