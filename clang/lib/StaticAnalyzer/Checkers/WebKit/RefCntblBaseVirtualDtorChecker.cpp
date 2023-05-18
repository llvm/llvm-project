//=======- RefCntblBaseVirtualDtor.cpp ---------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DiagOutputUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {
class RefCntblBaseVirtualDtorChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
private:
  BugType Bug;
  mutable BugReporter *BR;

public:
  RefCntblBaseVirtualDtorChecker()
      : Bug(this,
            "Reference-countable base class doesn't have virtual destructor",
            "WebKit coding guidelines") {}

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : public RecursiveASTVisitor<LocalVisitor> {
      const RefCntblBaseVirtualDtorChecker *Checker;
      explicit LocalVisitor(const RefCntblBaseVirtualDtorChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
      }

      bool shouldVisitTemplateInstantiations() const { return true; }
      bool shouldVisitImplicitCode() const { return false; }

      bool VisitCXXRecordDecl(const CXXRecordDecl *RD) {
        Checker->visitCXXRecordDecl(RD);
        return true;
      }
    };

    LocalVisitor visitor(this);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitCXXRecordDecl(const CXXRecordDecl *RD) const {
    if (shouldSkipDecl(RD))
      return;

    CXXBasePaths Paths;
    Paths.setOrigin(RD);

    const CXXBaseSpecifier *ProblematicBaseSpecifier = nullptr;
    const CXXRecordDecl *ProblematicBaseClass = nullptr;

    const auto IsPublicBaseRefCntblWOVirtualDtor =
        [RD, &ProblematicBaseSpecifier,
         &ProblematicBaseClass](const CXXBaseSpecifier *Base, CXXBasePath &) {
          const auto AccSpec = Base->getAccessSpecifier();
          if (AccSpec == AS_protected || AccSpec == AS_private ||
              (AccSpec == AS_none && RD->isClass()))
            return false;

          auto hasRefInBase = clang::hasPublicMethodInBase(Base, "ref");
          auto hasDerefInBase = clang::hasPublicMethodInBase(Base, "deref");

          bool hasRef = hasRefInBase && *hasRefInBase != nullptr;
          bool hasDeref = hasDerefInBase && *hasDerefInBase != nullptr;

          QualType T = Base->getType();
          if (T.isNull())
            return false;

          const CXXRecordDecl *C = T->getAsCXXRecordDecl();
          if (!C)
            return false;
          bool AnyInconclusiveBase = false;
          const auto hasPublicRefInBase =
              [&AnyInconclusiveBase](const CXXBaseSpecifier *Base,
                                     CXXBasePath &) {
                auto hasRefInBase = clang::hasPublicMethodInBase(Base, "ref");
                if (!hasRefInBase) {
                  AnyInconclusiveBase = true;
                  return false;
                }
                return (*hasRefInBase) != nullptr;
              };
          const auto hasPublicDerefInBase = [&AnyInconclusiveBase](
                                                const CXXBaseSpecifier *Base,
                                                CXXBasePath &) {
            auto hasDerefInBase = clang::hasPublicMethodInBase(Base, "deref");
            if (!hasDerefInBase) {
              AnyInconclusiveBase = true;
              return false;
            }
            return (*hasDerefInBase) != nullptr;
          };
          CXXBasePaths Paths;
          Paths.setOrigin(C);
          hasRef = hasRef || C->lookupInBases(hasPublicRefInBase, Paths,
                                              /*LookupInDependent =*/true);
          hasDeref = hasDeref || C->lookupInBases(hasPublicDerefInBase, Paths,
                                                  /*LookupInDependent =*/true);
          if (AnyInconclusiveBase || !hasRef || !hasDeref)
            return false;

          const auto *Dtor = C->getDestructor();
          if (!Dtor || !Dtor->isVirtual()) {
            ProblematicBaseSpecifier = Base;
            ProblematicBaseClass = C;
            return true;
          }

          return false;
        };

    if (RD->lookupInBases(IsPublicBaseRefCntblWOVirtualDtor, Paths,
                          /*LookupInDependent =*/true)) {
      reportBug(RD, ProblematicBaseSpecifier, ProblematicBaseClass);
    }
  }

  bool shouldSkipDecl(const CXXRecordDecl *RD) const {
    if (!RD->isThisDeclarationADefinition())
      return true;

    if (RD->isImplicit())
      return true;

    if (RD->isLambda())
      return true;

    // If the construct doesn't have a source file, then it's not something
    // we want to diagnose.
    const auto RDLocation = RD->getLocation();
    if (!RDLocation.isValid())
      return true;

    const auto Kind = RD->getTagKind();
    if (Kind != TagTypeKind::Struct && Kind != TagTypeKind::Class)
      return true;

    // Ignore CXXRecords that come from system headers.
    if (BR->getSourceManager().getFileCharacteristic(RDLocation) !=
        SrcMgr::C_User)
      return true;

    return false;
  }

  void reportBug(const CXXRecordDecl *DerivedClass,
                 const CXXBaseSpecifier *BaseSpec,
                 const CXXRecordDecl *ProblematicBaseClass) const {
    assert(DerivedClass);
    assert(BaseSpec);
    assert(ProblematicBaseClass);

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << (ProblematicBaseClass->isClass() ? "Class" : "Struct") << " ";
    printQuotedQualifiedName(Os, ProblematicBaseClass);

    Os << " is used as a base of "
       << (DerivedClass->isClass() ? "class" : "struct") << " ";
    printQuotedQualifiedName(Os, DerivedClass);

    Os << " but doesn't have virtual destructor";

    PathDiagnosticLocation BSLoc(BaseSpec->getSourceRange().getBegin(),
                                 BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(BaseSpec->getSourceRange());
    BR->emitReport(std::move(Report));
  }
};
} // namespace

void ento::registerRefCntblBaseVirtualDtorChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<RefCntblBaseVirtualDtorChecker>();
}

bool ento::shouldRegisterRefCntblBaseVirtualDtorChecker(
    const CheckerManager &mgr) {
  return true;
}
