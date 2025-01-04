//=======- RawPtrRefMemberChecker.cpp ----------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASTUtils.h"
#include "DiagOutputUtils.h"
#include "PtrTypesSemantics.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "llvm/Support/Casting.h"
#include <optional>

using namespace clang;
using namespace ento;

namespace {

class RawPtrRefMemberChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
private:
  BugType Bug;
  mutable BugReporter *BR;

public:
  RawPtrRefMemberChecker(const char *description)
      : Bug(this, description, "WebKit coding guidelines") {}

  virtual std::optional<bool>
  isPtrCompatible(const clang::CXXRecordDecl *) const = 0;
  virtual bool isPtrCls(const clang::CXXRecordDecl *) const = 0;
  virtual const char *typeName() const = 0;
  virtual const char *invariant() const = 0;

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    // The calls to checkAST* from AnalysisConsumer don't
    // visit template instantiations or lambda classes. We
    // want to visit those, so we make our own RecursiveASTVisitor.
    struct LocalVisitor : DynamicRecursiveASTVisitor {
      const RawPtrRefMemberChecker *Checker;
      explicit LocalVisitor(const RawPtrRefMemberChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
        ShouldVisitTemplateInstantiations = true;
        ShouldVisitImplicitCode = false;
      }

      bool VisitRecordDecl(RecordDecl *RD) override {
        Checker->visitRecordDecl(RD);
        return true;
      }
    };

    LocalVisitor visitor(this);
    visitor.TraverseDecl(const_cast<TranslationUnitDecl *>(TUD));
  }

  void visitRecordDecl(const RecordDecl *RD) const {
    if (shouldSkipDecl(RD))
      return;

    for (auto *Member : RD->fields()) {
      const Type *MemberType = Member->getType().getTypePtrOrNull();
      if (!MemberType)
        continue;

      if (auto *MemberCXXRD = MemberType->getPointeeCXXRecordDecl()) {
        // If we don't see the definition we just don't know.
        if (MemberCXXRD->hasDefinition()) {
          std::optional<bool> isRCAble = isPtrCompatible(MemberCXXRD);
          if (isRCAble && *isRCAble)
            reportBug(Member, MemberType, MemberCXXRD, RD);
        }
      }
    }
  }

  bool shouldSkipDecl(const RecordDecl *RD) const {
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
    // FIMXE: Should we check union members too?
    if (Kind != TagTypeKind::Struct && Kind != TagTypeKind::Class)
      return true;

    // Ignore CXXRecords that come from system headers.
    if (BR->getSourceManager().isInSystemHeader(RDLocation))
      return true;

    // Ref-counted smartpointers actually have raw-pointer to uncounted type as
    // a member but we trust them to handle it correctly.
    auto CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(RD);
    if (CXXRD)
      return isPtrCls(CXXRD);

    return false;
  }

  void reportBug(const FieldDecl *Member, const Type *MemberType,
                 const CXXRecordDecl *MemberCXXRD,
                 const RecordDecl *ClassCXXRD) const {
    assert(Member);
    assert(MemberType);
    assert(MemberCXXRD);

    SmallString<100> Buf;
    llvm::raw_svector_ostream Os(Buf);

    Os << "Member variable ";
    printQuotedName(Os, Member);
    Os << " in ";
    printQuotedQualifiedName(Os, ClassCXXRD);
    Os << " is a "
       << (isa<PointerType>(MemberType) ? "raw pointer" : "reference") << " to "
       << typeName() << " ";
    printQuotedQualifiedName(Os, MemberCXXRD);
    Os << "; " << invariant() << ".";

    PathDiagnosticLocation BSLoc(Member->getSourceRange().getBegin(),
                                 BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Os.str(), BSLoc);
    Report->addRange(Member->getSourceRange());
    BR->emitReport(std::move(Report));
  }
};

class NoUncountedMemberChecker final : public RawPtrRefMemberChecker {
public:
  NoUncountedMemberChecker()
      : RawPtrRefMemberChecker("Member variable is a raw-pointer/reference to "
                               "reference-countable type") {}

  std::optional<bool>
  isPtrCompatible(const clang::CXXRecordDecl *R) const final {
    return isRefCountable(R);
  }

  bool isPtrCls(const clang::CXXRecordDecl *R) const final {
    return isRefCounted(R);
  }

  const char *typeName() const final { return "ref-countable type"; }

  const char *invariant() const final {
    return "member variables must be Ref, RefPtr, WeakRef, or WeakPtr";
  }
};

class NoUncheckedPtrMemberChecker final : public RawPtrRefMemberChecker {
public:
  NoUncheckedPtrMemberChecker()
      : RawPtrRefMemberChecker("Member variable is a raw-pointer/reference to "
                               "checked-pointer capable type") {}

  std::optional<bool>
  isPtrCompatible(const clang::CXXRecordDecl *R) const final {
    return isCheckedPtrCapable(R);
  }

  bool isPtrCls(const clang::CXXRecordDecl *R) const final {
    return isCheckedPtr(R);
  }

  const char *typeName() const final { return "CheckedPtr capable type"; }

  const char *invariant() const final {
    return "member variables must be a CheckedPtr, CheckedRef, WeakRef, or "
           "WeakPtr";
  }
};

} // namespace

void ento::registerNoUncountedMemberChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<NoUncountedMemberChecker>();
}

bool ento::shouldRegisterNoUncountedMemberChecker(const CheckerManager &Mgr) {
  return true;
}

void ento::registerNoUncheckedPtrMemberChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<NoUncheckedPtrMemberChecker>();
}

bool ento::shouldRegisterNoUncheckedPtrMemberChecker(
    const CheckerManager &Mgr) {
  return true;
}
