//=======- BorrowPlacementChecker.cpp ----------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A Borrow<T> saves its object's borrowed flag on construction and restores it
// on destruction, so Borrows of one object must be destroyed in the reverse
// order of their construction. The language guarantees that order only for
// local variables, parameters and temporaries; this checker reports a Borrow
// that lives anywhere else.
//
//===----------------------------------------------------------------------===//

#include "PtrTypesSemantics.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;
using namespace ento;

namespace {

bool ownsBorrow(QualType T, llvm::SmallPtrSetImpl<const CXXRecordDecl *> &Seen);

bool templateArgOwnsBorrow(const TemplateArgument &Arg,
                           llvm::SmallPtrSetImpl<const CXXRecordDecl *> &Seen) {
  if (Arg.getKind() == TemplateArgument::Type)
    return ownsBorrow(Arg.getAsType(), Seen);
  if (Arg.getKind() == TemplateArgument::Pack) {
    for (const auto &Element : Arg.pack_elements()) {
      if (templateArgOwnsBorrow(Element, Seen))
        return true;
    }
  }
  return false;
}

// Pointers and references don't own their pointee, but a template argument
// does count: std::unique_ptr<Borrow<T>> holds no Borrow member yet owns one.
bool ownsBorrow(QualType T,
                llvm::SmallPtrSetImpl<const CXXRecordDecl *> &Seen) {
  if (T.isNull())
    return false;
  const auto *Record = T->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();
  if (!Record)
    return false;
  if (isBorrow(Record))
    return true;
  if (!Seen.insert(Record).second)
    return false;
  if (const auto *Specialization =
          dyn_cast<ClassTemplateSpecializationDecl>(Record)) {
    for (const auto &Arg : Specialization->getTemplateArgs().asArray()) {
      if (templateArgOwnsBorrow(Arg, Seen))
        return true;
    }
  }
  Record = Record->getDefinition();
  if (!Record)
    return false;
  for (const auto &Base : Record->bases()) {
    if (ownsBorrow(Base.getType(), Seen))
      return true;
  }
  for (const auto *Field : Record->fields()) {
    if (ownsBorrow(Field->getType(), Seen))
      return true;
  }
  return false;
}

bool ownsBorrow(QualType T) {
  llvm::SmallPtrSet<const CXXRecordDecl *, 8> Seen;
  return ownsBorrow(T, Seen);
}

bool isBorrowOrArrayOfBorrow(QualType T) {
  return !T.isNull() &&
         isBorrow(T->getBaseElementTypeUnsafe()->getAsCXXRecordDecl());
}

bool isCoroutine(const DeclContext *DC) {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(DC);
  return FD && isa_and_nonnull<CoroutineBodyStmt>(FD->getBody());
}

bool isStdLifetimeFunction(const FunctionDecl *FD) {
  if (!FD || !FD->isInStdNamespace() || !FD->getIdentifier())
    return false;
  return llvm::StringSwitch<bool>(FD->getName())
      .Cases({"construct_at", "destroy_at", "destroy", "destroy_n"}, true)
      .Default(false);
}

class BorrowPlacementChecker
    : public Checker<check::ASTDecl<TranslationUnitDecl>> {
  BugType Bug;
  mutable BugReporter *BR = nullptr;

public:
  BorrowPlacementChecker()
      : Bug(this, "Borrow stored where it can be destroyed out of order",
            "WebKit coding guidelines") {}

  void checkASTDecl(const TranslationUnitDecl *TUD, AnalysisManager &MGR,
                    BugReporter &BRArg) const {
    BR = &BRArg;

    struct LocalVisitor final : public ConstDynamicRecursiveASTVisitor {
      const BorrowPlacementChecker *Checker;
      const Decl *DeclWithIssue = nullptr;

      explicit LocalVisitor(const BorrowPlacementChecker *Checker)
          : Checker(Checker) {
        assert(Checker);
        ShouldVisitTemplateInstantiations = true;
        ShouldVisitImplicitCode = false;
        ShouldVisitLambdaBody = true;
      }

      // Template patterns are checked through their instantiations.
      bool TraverseDecl(const Decl *D) override {
        if (D && !isa<TemplateDecl>(D) && D->isTemplated())
          return true;
        llvm::SaveAndRestore SavedDecl(DeclWithIssue);
        if (isa_and_nonnull<FunctionDecl, VarDecl, FieldDecl>(D))
          DeclWithIssue = D;
        return ConstDynamicRecursiveASTVisitor::TraverseDecl(D);
      }

      bool VisitVarDecl(const VarDecl *V) override {
        Checker->visitVarDecl(V);
        return true;
      }

      bool VisitFieldDecl(const FieldDecl *F) override {
        Checker->visitFieldDecl(F);
        return true;
      }

      bool VisitFunctionDecl(const FunctionDecl *FD) override {
        Checker->visitFunctionDecl(FD);
        return true;
      }

      bool VisitCXXNewExpr(const CXXNewExpr *E) override {
        Checker->visitCXXNewExpr(E, DeclWithIssue);
        return true;
      }

      bool VisitMaterializeTemporaryExpr(
          const MaterializeTemporaryExpr *E) override {
        Checker->visitMaterializeTemporaryExpr(E, DeclWithIssue);
        return true;
      }

      bool VisitCallExpr(const CallExpr *E) override {
        Checker->visitCallExpr(E, DeclWithIssue);
        return true;
      }
    };

    LocalVisitor Visitor(this);
    Visitor.TraverseDecl(TUD);
  }

  bool isInSystemHeader(SourceLocation Loc) const {
    return BR->getSourceManager().isInSystemHeader(Loc);
  }

  void visitVarDecl(const VarDecl *V) const {
    if (isInSystemHeader(V->getLocation()))
      return;
    QualType T = V->getType();
    if (T->isReferenceType())
      return;

    if (!isBorrowOrArrayOfBorrow(T)) {
      if (ownsBorrow(T))
        reportWrapped(V, V->isInitCapture() ? "Lambda capture" : "Variable", T,
                      V);
      return;
    }

    if (V->isInitCapture())
      report(V, "Lambda capture", "is stored in the closure object", V);
    else if (!V->hasLocalStorage())
      report(V, "Variable", "has static or thread storage duration", V);
    else if (isCoroutine(V->getParentFunctionOrMethod()))
      report(V, "Variable", "lives in a coroutine frame", V);
  }

  void visitFieldDecl(const FieldDecl *F) const {
    if (isInSystemHeader(F->getLocation()))
      return;
    if (!ownsBorrow(F->getType()))
      return;
    if (isBorrowOrArrayOfBorrow(F->getType()))
      report(F, "Member", "is a Borrow", F);
    else
      reportWrapped(F, "Member", F->getType(), F);
  }

  void visitFunctionDecl(const FunctionDecl *FD) const {
    if (isInSystemHeader(FD->getLocation()))
      return;
    QualType T = FD->getReturnType();
    if (isBorrowOrArrayOfBorrow(T) || !ownsBorrow(T))
      return;
    reportWrapped(FD, "Return value of", T, FD);
  }

  void visitCXXNewExpr(const CXXNewExpr *E, const Decl *DeclWithIssue) const {
    if (isInSystemHeader(E->getBeginLoc()))
      return;
    QualType T = E->getAllocatedType();
    if (!ownsBorrow(T))
      return;
    reportExpr(E, DeclWithIssue, [&](llvm::raw_ostream &Os) {
      Os << "New expression creates a Borrow";
      if (!isBorrowOrArrayOfBorrow(T))
        Os << " inside " << quoted(T);
      Os << "; a Borrow may only be a local variable, parameter, or "
            "temporary";
    });
  }

  void visitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *E,
                                     const Decl *DeclWithIssue) const {
    if (isInSystemHeader(E->getBeginLoc()))
      return;
    QualType T = E->getType();
    if (isBorrowOrArrayOfBorrow(T) || !ownsBorrow(T))
      return;
    reportExpr(E, DeclWithIssue, [&](llvm::raw_ostream &Os) {
      Os << "Temporary stores a Borrow inside " << quoted(T)
         << "; a Borrow may only be a local variable, parameter, or "
            "temporary";
    });
  }

  void visitCallExpr(const CallExpr *E, const Decl *DeclWithIssue) const {
    if (isInSystemHeader(E->getBeginLoc()))
      return;

    if (const auto *Member = dyn_cast<CXXMemberCallExpr>(E)) {
      if (isa_and_nonnull<CXXDestructorDecl>(Member->getMethodDecl()) &&
          isBorrow(Member->getRecordDecl())) {
        reportExpr(E, DeclWithIssue, [&](llvm::raw_ostream &Os) {
          Os << "Explicit destructor call ends a Borrow out of order";
        });
      }
      return;
    }

    if (!isStdLifetimeFunction(E->getDirectCallee()))
      return;
    for (const auto *Arg : E->arguments()) {
      QualType T = Arg->getType();
      if (!T->isPointerType() || !ownsBorrow(T->getPointeeType()))
        continue;
      reportExpr(E, DeclWithIssue, [&](llvm::raw_ostream &Os) {
        Os << "Call to " << quoted(E->getDirectCallee())
           << " manages the lifetime of a Borrow; a Borrow may only be a "
              "local variable, parameter, or temporary";
      });
      return;
    }
  }

private:
  std::string quoted(QualType T) const {
    return "'" +
           T.getUnqualifiedType().getAsString(
               BR->getContext().getPrintingPolicy()) +
           "'";
  }

  static std::string quoted(const FunctionDecl *FD) {
    return "'" + FD->getQualifiedNameAsString() + "'";
  }

  void reportWrapped(const NamedDecl *D, StringRef What, QualType T,
                     const Decl *DeclWithIssue) const {
    SmallString<160> Buf;
    llvm::raw_svector_ostream Os(Buf);
    Os << What << " '" << D->getName() << "' stores a Borrow inside "
       << quoted(T)
       << "; a Borrow may only be a local variable, parameter, or temporary";
    emit(Os.str(), D->getLocation(), D->getSourceRange(), DeclWithIssue);
  }

  void report(const NamedDecl *D, StringRef What, StringRef Why,
              const Decl *DeclWithIssue) const {
    SmallString<160> Buf;
    llvm::raw_svector_ostream Os(Buf);
    Os << What << " '" << D->getName() << "' " << Why
       << "; a Borrow may only be a local variable, parameter, or temporary";
    emit(Os.str(), D->getLocation(), D->getSourceRange(), DeclWithIssue);
  }

  template <typename Describe>
  void reportExpr(const Expr *E, const Decl *DeclWithIssue,
                  Describe describe) const {
    SmallString<160> Buf;
    llvm::raw_svector_ostream Os(Buf);
    describe(Os);
    emit(Os.str(), E->getExprLoc(), E->getSourceRange(), DeclWithIssue);
  }

  void emit(StringRef Message, SourceLocation Loc, SourceRange Range,
            const Decl *DeclWithIssue) const {
    PathDiagnosticLocation BSLoc(Loc, BR->getSourceManager());
    auto Report = std::make_unique<BasicBugReport>(Bug, Message, BSLoc);
    Report->addRange(Range);
    if (DeclWithIssue)
      Report->setDeclWithIssue(DeclWithIssue);
    BR->emitReport(std::move(Report));
  }
};

} // namespace

void ento::registerBorrowPlacementChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<BorrowPlacementChecker>();
}

bool ento::shouldRegisterBorrowPlacementChecker(const CheckerManager &) {
  return true;
}
