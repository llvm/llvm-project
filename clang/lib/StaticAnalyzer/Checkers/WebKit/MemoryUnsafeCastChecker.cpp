//=======- MemoryUnsafeCastChecker.cpp -------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MemoryUnsafeCast checker, which checks for casts from a
// base type to a derived type.
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"

using namespace clang;
using namespace ento;
using namespace ast_matchers;

namespace {
static constexpr const char *const BaseNode = "BaseNode";
static constexpr const char *const DerivedNode = "DerivedNode";
static constexpr const char *const FromCastNode = "FromCast";
static constexpr const char *const ToCastNode = "ToCast";
static constexpr const char *const WarnRecordDecl = "WarnRecordDecl";

class MemoryUnsafeCastChecker : public Checker<check::ASTCodeBody> {
  BugType BT{this, "Unsafe cast", "WebKit coding guidelines"};

public:
  void checkASTCodeBody(const Decl *D, AnalysisManager &Mgr,
                        BugReporter &BR) const;
};
} // end namespace

static void emitDiagnostics(const BoundNodes &Nodes, BugReporter &BR,
                            AnalysisDeclContext *ADC,
                            const MemoryUnsafeCastChecker *Checker,
                            const BugType &BT) {
  const auto *CE = Nodes.getNodeAs<CastExpr>(WarnRecordDecl);
  const NamedDecl *Base = Nodes.getNodeAs<NamedDecl>(BaseNode);
  const NamedDecl *Derived = Nodes.getNodeAs<NamedDecl>(DerivedNode);
  assert(CE && Base && Derived);

  std::string Diagnostics;
  llvm::raw_string_ostream OS(Diagnostics);
  OS << "Unsafe cast from base type '" << Base->getNameAsString()
     << "' to derived type '" << Derived->getNameAsString() << "'";
  PathDiagnosticLocation BSLoc(CE->getSourceRange().getBegin(),
                               BR.getSourceManager());
  auto Report = std::make_unique<BasicBugReport>(BT, OS.str(), BSLoc);
  Report->addRange(CE->getSourceRange());
  Report->setDeclWithIssue(ADC->getDecl());
  BR.emitReport(std::move(Report));
}

static void emitDiagnosticsUnrelated(const BoundNodes &Nodes, BugReporter &BR,
                                     AnalysisDeclContext *ADC,
                                     const MemoryUnsafeCastChecker *Checker,
                                     const BugType &BT) {
  const auto *CE = Nodes.getNodeAs<CastExpr>(WarnRecordDecl);
  const NamedDecl *FromCast = Nodes.getNodeAs<NamedDecl>(FromCastNode);
  const NamedDecl *ToCast = Nodes.getNodeAs<NamedDecl>(ToCastNode);
  assert(CE && FromCast && ToCast);

  std::string Diagnostics;
  llvm::raw_string_ostream OS(Diagnostics);
  OS << "Unsafe cast from type '" << FromCast->getNameAsString()
     << "' to an unrelated type '" << ToCast->getNameAsString() << "'";
  PathDiagnosticLocation BSLoc(CE->getSourceRange().getBegin(),
                               BR.getSourceManager());
  auto Report = std::make_unique<BasicBugReport>(BT, OS.str(), BSLoc);
  Report->addRange(CE->getSourceRange());
  Report->setDeclWithIssue(ADC->getDecl());
  BR.emitReport(std::move(Report));
}

static void emitDiagnosticsIdArg(const BoundNodes &Nodes, BugReporter &BR,
                                 AnalysisDeclContext *ADC,
                                 const MemoryUnsafeCastChecker *Checker,
                                 const BugType &BT) {
  const auto *CE = Nodes.getNodeAs<CastExpr>(WarnRecordDecl);
  const NamedDecl *Derived = Nodes.getNodeAs<NamedDecl>(DerivedNode);
  assert(CE && Derived);

  std::string Diagnostics;
  llvm::raw_string_ostream OS(Diagnostics);
  OS << "Unsafe implicit cast from 'id' to specific type '"
     << Derived->getNameAsString() << "'";
  PathDiagnosticLocation BSLoc(CE->getSourceRange().getBegin(),
                               BR.getSourceManager());
  auto Report = std::make_unique<BasicBugReport>(BT, OS.str(), BSLoc);
  Report->addRange(CE->getSourceRange());
  Report->setDeclWithIssue(ADC->getDecl());
  BR.emitReport(std::move(Report));
}

namespace {
using BoundNodesMap = ::clang::ast_matchers::internal::BoundNodesMap;

// Matches the plain `id` type.
AST_MATCHER(QualType, isObjCIdType) { return Node->isObjCIdType(); }

// Matches a cast whose previously-bound BaseID node is a class template
// specialization and whose previously-bound DerivedID node is one of that
// specialization's type template arguments, i.e. the CRTP pattern
// `class Derived : Base<Derived>`.
AST_MATCHER_P2(Expr, isCRTPCast, std::string, BaseID, std::string, DerivedID) {
  return Builder->removeBindings([this](const BoundNodesMap &Nodes) {
    const auto *Base = Nodes.getNodeAs<CXXRecordDecl>(this->BaseID);
    const auto *Derived = Nodes.getNodeAs<CXXRecordDecl>(this->DerivedID);
    const auto *CTSD =
        Base ? dyn_cast<ClassTemplateSpecializationDecl>(Base) : nullptr;
    if (!CTSD || !Derived)
      return true;
    for (const TemplateArgument &Arg : CTSD->getTemplateArgs().asArray()) {
      if (Arg.getKind() != TemplateArgument::Type)
        continue;
      QualType ArgType = Arg.getAsType();
      if (!ArgType.isNull() && ArgType->getAsCXXRecordDecl() == Derived)
        return false;
    }
    return true;
  });
}
} // end anonymous namespace

static decltype(auto) hasTypePointingTo(DeclarationMatcher DeclM) {
  return hasType(pointerType(pointee(hasDeclaration(DeclM))));
}

// Matches `this` or `*this`, but not member accesses like `this->m_field`.
static decltype(auto) isThisOrDerefThis() {
  return ignoringParenImpCasts(anyOf(
      cxxThisExpr(),
      unaryOperator(hasOperatorName("*"),
                    hasUnaryOperand(ignoringParenImpCasts(cxxThisExpr())))));
}

void MemoryUnsafeCastChecker::checkASTCodeBody(const Decl *D,
                                               AnalysisManager &AM,
                                               BugReporter &BR) const {

  AnalysisDeclContext *ADC = AM.getAnalysisDeclContext(D);

  // Match downcasts from base type to derived type and warn
  auto MatchExprPtr = allOf(
      hasSourceExpression(hasTypePointingTo(cxxRecordDecl().bind(BaseNode))),
      hasTypePointingTo(cxxRecordDecl(isDerivedFrom(equalsBoundNode(BaseNode)))
                            .bind(DerivedNode)),
      unless(anyOf(hasTypePointingTo(templateTypeParmDecl()),
                   allOf(hasSourceExpression(cxxThisExpr()),
                         isCRTPCast(BaseNode, DerivedNode)))));
  auto MatchExprPtrObjC = allOf(
      hasSourceExpression(ignoringImpCasts(hasType(objcObjectPointerType(
          pointee(hasDeclaration(objcInterfaceDecl().bind(BaseNode))))))),
      ignoringImpCasts(hasType(objcObjectPointerType(pointee(hasDeclaration(
          objcInterfaceDecl(isDerivedFrom(equalsBoundNode(BaseNode)))
              .bind(DerivedNode)))))));
  auto MatchExprRefTypeDef =
      allOf(hasSourceExpression(hasType(hasUnqualifiedDesugaredType(recordType(
                hasDeclaration(decl(cxxRecordDecl().bind(BaseNode))))))),
            hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
                decl(cxxRecordDecl(isDerivedFrom(equalsBoundNode(BaseNode)))
                         .bind(DerivedNode)))))),
            unless(anyOf(hasType(templateTypeParmDecl()),
                         allOf(hasSourceExpression(isThisOrDerefThis()),
                               isCRTPCast(BaseNode, DerivedNode)))));
  auto MatchExprPtrVoidCast = allOf(
      anyOf(hasSourceExpression(explicitCastExpr(
                hasType(pointerType(pointee(voidType()))),
                hasSourceExpression(ignoringImpCasts(
                    hasTypePointingTo(cxxRecordDecl().bind(BaseNode)))))),
            hasSourceExpression(
                callExpr(hasType(pointerType(pointee(voidType()))),
                         hasAnyArgument(ignoringImpCasts(hasTypePointingTo(
                             cxxRecordDecl().bind(BaseNode))))))),
      hasTypePointingTo(cxxRecordDecl(isDerivedFrom(equalsBoundNode(BaseNode)))
                            .bind(DerivedNode)));

  auto ExplicitCast =
      explicitCastExpr(anyOf(MatchExprPtr, MatchExprRefTypeDef,
                             MatchExprPtrObjC, MatchExprPtrVoidCast))
          .bind(WarnRecordDecl);
  auto Cast = stmt(ExplicitCast);

  auto Matches =
      match(stmt(forEachDescendant(Cast)), *D->getBody(), AM.getASTContext());
  for (BoundNodes Match : Matches)
    emitDiagnostics(Match, BR, ADC, this, BT);

  // Match calls returning derived type where an argument is a void pointer.
  auto VoidPtrCast =
      castExpr(hasType(pointerType(pointee(voidType()))),
               hasSourceExpression(ignoringImpCasts(
                   hasTypePointingTo(cxxRecordDecl().bind(BaseNode)))))
          .bind(WarnRecordDecl);
  auto MatchCallPtrVoidArgCast = callExpr(
      hasAnyArgument(anyOf(VoidPtrCast,
                           explicitCastExpr(hasSourceExpression(VoidPtrCast)))),
      hasTypePointingTo(cxxRecordDecl(isDerivedFrom(equalsBoundNode(BaseNode)))
                            .bind(DerivedNode)));
  auto CallArgCast = stmt(MatchCallPtrVoidArgCast);
  auto MatchesCallArgCast = match(stmt(forEachDescendant(CallArgCast)),
                                  *D->getBody(), AM.getASTContext());
  for (BoundNodes Match : MatchesCallArgCast)
    emitDiagnostics(Match, BR, ADC, this, BT);

  // Match casts between unrelated types and warn
  auto MatchExprPtrUnrelatedTypes = allOf(
      hasSourceExpression(
          hasTypePointingTo(cxxRecordDecl().bind(FromCastNode))),
      hasTypePointingTo(cxxRecordDecl().bind(ToCastNode)),
      unless(anyOf(hasTypePointingTo(cxxRecordDecl(
                       isSameOrDerivedFrom(equalsBoundNode(FromCastNode)))),
                   hasSourceExpression(hasTypePointingTo(cxxRecordDecl(
                       isSameOrDerivedFrom(equalsBoundNode(ToCastNode))))))));
  auto MatchExprPtrObjCUnrelatedTypes = allOf(
      hasSourceExpression(ignoringImpCasts(hasType(objcObjectPointerType(
          pointee(hasDeclaration(objcInterfaceDecl().bind(FromCastNode))))))),
      ignoringImpCasts(hasType(objcObjectPointerType(
          pointee(hasDeclaration(objcInterfaceDecl().bind(ToCastNode)))))),
      unless(anyOf(
          ignoringImpCasts(hasType(
              objcObjectPointerType(pointee(hasDeclaration(objcInterfaceDecl(
                  isSameOrDerivedFrom(equalsBoundNode(FromCastNode)))))))),
          hasSourceExpression(ignoringImpCasts(hasType(
              objcObjectPointerType(pointee(hasDeclaration(objcInterfaceDecl(
                  isSameOrDerivedFrom(equalsBoundNode(ToCastNode))))))))))));
  auto MatchExprRefTypeDefUnrelated = allOf(
      hasSourceExpression(hasType(hasUnqualifiedDesugaredType(recordType(
          hasDeclaration(decl(cxxRecordDecl().bind(FromCastNode))))))),
      hasType(hasUnqualifiedDesugaredType(
          recordType(hasDeclaration(decl(cxxRecordDecl().bind(ToCastNode)))))),
      unless(anyOf(
          hasType(hasUnqualifiedDesugaredType(
              recordType(hasDeclaration(decl(cxxRecordDecl(
                  isSameOrDerivedFrom(equalsBoundNode(FromCastNode)))))))),
          hasSourceExpression(hasType(hasUnqualifiedDesugaredType(
              recordType(hasDeclaration(decl(cxxRecordDecl(
                  isSameOrDerivedFrom(equalsBoundNode(ToCastNode))))))))))));

  auto ExplicitCastUnrelated =
      explicitCastExpr(anyOf(MatchExprPtrUnrelatedTypes,
                             MatchExprPtrObjCUnrelatedTypes,
                             MatchExprRefTypeDefUnrelated))
          .bind(WarnRecordDecl);
  auto CastUnrelated = stmt(ExplicitCastUnrelated);
  auto MatchesUnrelatedTypes = match(stmt(forEachDescendant(CastUnrelated)),
                                     *D->getBody(), AM.getASTContext());
  for (BoundNodes Match : MatchesUnrelatedTypes)
    emitDiagnosticsUnrelated(Match, BR, ADC, this, BT);

  // Match an `id`-typed argument implicitly converted to a specific
  // Objective-C type at a call, message send, or constructor call, e.g.
  // passing an `id` where an `NSString *` parameter is expected. Such
  // conversions compile without a visible cast but throw at runtime if the
  // object is not actually of that type.
  auto CastArgFromIdToSpecificType =
      implicitCastExpr(
          hasCastKind(CK_BitCast),
          hasSourceExpression(
              ignoringParenImpCasts(hasType(qualType(isObjCIdType())))),
          hasType(qualType(hasCanonicalType(objcObjectPointerType(pointee(
              hasDeclaration(objcInterfaceDecl().bind(DerivedNode))))))))
          .bind(WarnRecordDecl);
  auto MatchCallArgFromId =
      anyOf(callExpr(hasAnyArgument(CastArgFromIdToSpecificType)),
            cxxConstructExpr(hasAnyArgument(CastArgFromIdToSpecificType)),
            objcMessageExpr(hasAnyArgument(CastArgFromIdToSpecificType)));
  auto MatchesCallArgFromId =
      match(stmt(forEachDescendant(stmt(MatchCallArgFromId))), *D->getBody(),
            AM.getASTContext());
  for (BoundNodes Match : MatchesCallArgFromId)
    emitDiagnosticsIdArg(Match, BR, ADC, this, BT);
}

void ento::registerMemoryUnsafeCastChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<MemoryUnsafeCastChecker>();
}

bool ento::shouldRegisterMemoryUnsafeCastChecker(const CheckerManager &mgr) {
  return true;
}
