//=======- RawPtrRefSafetyModel.cpp -----------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RawPtrRefSafetyModel.h"
#include "ASTUtils.h"
#include "DiagOutputUtils.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/DomainSpecific/CocoaConventions.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;

namespace {

class RefCountedSafetyModel : public PtrRefSafetyModel {
public:
  std::optional<bool> isUnsafeType(QualType QT) const override {
    return isUncounted(QT);
  }
  std::optional<bool> isUnsafePtr(QualType QT, bool) const override {
    return isUncountedPtr(QT.getCanonicalType());
  }
  bool isSafePtr(const CXXRecordDecl *Record) const override {
    return isRefCounted(Record) || isCheckedPtr(Record);
  }
  bool isSafePtrType(QualType T) const override {
    return isRefOrCheckedPtrType(T);
  }
  bool isPtrType(const std::string &Name) const override {
    return isRefType(Name);
  }
  const char *typeName() const override { return "RefPtr-capable type"; }
};

class CheckedPtrSafetyModel : public PtrRefSafetyModel {
public:
  std::optional<bool> isUnsafeType(QualType QT) const override {
    return isUnchecked(QT);
  }
  std::optional<bool> isUnsafePtr(QualType QT, bool) const override {
    return isUncheckedPtr(QT.getCanonicalType());
  }
  bool isSafePtr(const CXXRecordDecl *Record) const override {
    return isRefCounted(Record) || isCheckedPtr(Record);
  }
  bool isSafePtrType(QualType T) const override {
    return isRefOrCheckedPtrType(T);
  }
  bool isPtrType(const std::string &Name) const override {
    return isCheckedPtr(Name);
  }
  bool isSafeExpr(const Expr *E, bool, QualType, bool) const override {
    return isExprToGetCheckedPtrCapableMember(E);
  }
  const char *typeName() const override { return "CheckedPtr-capable type"; }
};

class RetainPtrSafetyModel : public PtrRefSafetyModel {
  mutable RetainTypeChecker RTC;

public:
  std::optional<bool> isUnsafeType(QualType QT) const override {
    return RTC.isUnretained(QT);
  }
  std::optional<bool> isUnsafePtr(QualType QT, bool IgnoreARC) const override {
    return RTC.isUnretained(QT, IgnoreARC);
  }
  bool isSafePtr(const CXXRecordDecl *Record) const override {
    return isRetainPtrOrOSPtr(Record);
  }
  bool isSafePtrType(QualType T) const override {
    return isRetainPtrOrOSPtrType(T);
  }
  bool isPtrType(const std::string &Name) const override {
    return isRetainPtrOrOSPtr(Name);
  }
  bool isSafeExpr(const Expr *E, bool, QualType, bool) const override {
    return ento::cocoa::isCocoaObjectRef(E->getType()) &&
           isa<ObjCMessageExpr>(E);
  }
  bool isSafeDecl(const Decl *D, const SourceManager &SM) const override {
    // Treat NS/CF globals in system header as immortal.
    return SM.isInSystemHeader(D->getLocation());
  }
  void describeHazard(llvm::raw_ostream &Os, const Expr *Origin,
                      QualType SinkType) const override {
    auto *VarType = SinkType.getTypePtr();
    if (isa<TypedefType>(VarType)) {
      Os << typeName() << " ";
      if (auto *Decl = RTC.getCanonicalDecl(SinkType)) {
        printQuotedQualifiedName(Os, Decl);
      } else {
        const auto *Typedef = VarType->getAs<TypedefType>();
        assert(Typedef);
        printQuotedQualifiedName(Os, Typedef->getDecl());
      }
      return;
    }
    PtrRefSafetyModel::describeHazard(Os, Origin, SinkType);
  }
  const char *typeName() const override { return "RetainPtr-capable type"; }
  RetainTypeChecker *retainTypeChecker() const override { return &RTC; }
};

static std::optional<bool> isCanBorrowType(QualType T) {
  auto *Record = T->getAsCXXRecordDecl();
  if (!Record)
    return false;
  return isBorrowable(Record);
}

static bool isSameRecord(QualType A, QualType B) {
  auto *RecordA = A->getAsCXXRecordDecl();
  auto *RecordB = B->getAsCXXRecordDecl();
  return RecordA && RecordB &&
         RecordA->getCanonicalDecl() == RecordB->getCanonicalDecl();
}

static const ValueDecl *declaredObject(const Expr *E) {
  const Expr *Stripped = E->IgnoreParenImpCasts();
  if (auto *DeclRef = dyn_cast<DeclRefExpr>(Stripped))
    return DeclRef->getDecl();
  if (auto *Member = dyn_cast<MemberExpr>(Stripped))
    return Member->getMemberDecl();
  return nullptr;
}

// Returns the const-declared variable or member that E denotes, directly or
// through a dereference that vends it as const, or nullptr.
static const ValueDecl *constDeclaredObject(const Expr *E) {
  const Expr *Stripped = E->IgnoreParenImpCasts();

  if (auto *Op = dyn_cast<CXXOperatorCallExpr>(Stripped)) {
    OverloadedOperatorKind Kind = Op->getOperator();
    if ((Kind != OO_Star && Kind != OO_Arrow) || Op->getNumArgs() != 1)
      return nullptr;
    if (!pointeeType(Op->getType()).isConstQualified())
      return nullptr;
    return constDeclaredObject(Op->getArg(0));
  }

  const ValueDecl *Decl = declaredObject(Stripped);
  if (!isa_and_nonnull<VarDecl>(Decl) && !isa_and_nonnull<FieldDecl>(Decl))
    return nullptr;
  QualType T = Decl->getType();
  if (!T.isConstQualified() || T->isReferenceType() || T->isPointerType())
    return nullptr;
  return Decl;
}

static bool mayHoldPointerTo(QualType ViewType, QualType CanBorrowType) {
  if (ViewType.isNull() || CanBorrowType.isNull())
    return false;

  if (QualType Pointee = ViewType->getPointeeType(); !Pointee.isNull())
    return isSameRecord(Pointee, CanBorrowType);

  auto *Record = ViewType->getAsCXXRecordDecl();
  if (!Record)
    return false;
  if (isBorrow(Record))
    return isSameRecord(borrowedType(ViewType), CanBorrowType);
  Record = Record->getDefinition();
  if (!Record)
    return true;

  return llvm::any_of(Record->fields(),
                      [&](const FieldDecl *Field) {
                        return mayHoldPointerTo(Field->getType(),
                                                CanBorrowType);
                      }) ||
         llvm::any_of(Record->bases(), [&](const CXXBaseSpecifier &Base) {
           return mayHoldPointerTo(Base.getType(), CanBorrowType);
         });
}

class BorrowSafetyModel : public PtrRefSafetyModel {
public:
  std::optional<bool> isUnsafeType(QualType QT) const override {
    return isView(QT);
  }
  std::optional<bool> isUnsafePtr(QualType QT, bool) const override {
    return isView(QT);
  }
  bool isSafePtr(const CXXRecordDecl *Record) const override {
    return isBorrow(Record);
  }
  bool isSafePtrType(QualType T) const override { return isBorrowType(T); }
  bool isPtrType(const std::string &Name) const override {
    return isBorrow(Name);
  }

  bool isSafeExpr(const Expr *Origin, bool PtrIsLifetimeBoundToOrigin,
                  QualType SinkType, bool SinkMayEscape) const override {
    QualType OriginType = pointeeType(Origin->getType());
    if (OriginType.isNull())
      return true;

    // A Borrow or a non-global const object guards a loan only within its own
    // scope, so neither vouches for a sink that may outlive that scope. This
    // holds for a reference to the Borrow itself, not just a loan through it.
    if (isBorrowType(OriginType))
      return !SinkMayEscape;

    if (!PtrIsLifetimeBoundToOrigin)
      return true;

    if (const ValueDecl *ConstObject = constDeclaredObject(Origin)) {
      auto *ConstVar = dyn_cast<VarDecl>(ConstObject);
      if (!SinkMayEscape || (ConstVar && ConstVar->hasGlobalStorage()))
        return true;
    }

    if (Origin->isPRValue() &&
        isCanBorrowType(Origin->getType()).value_or(false) &&
        !SinkType.isNull() && !mayHoldPointerTo(SinkType, Origin->getType()))
      return true;

    auto *Record = OriginType->getAsCXXRecordDecl();
    if (!Record)
      return true;

    auto Borrowable = isBorrowable(Record);
    return !Borrowable || !*Borrowable;
  }

  bool checksForInteriorDestruction() const override { return true; }
  bool recognizesIndirectStores() const override { return true; }
  const char *typeName() const override { return "CanBorrow type"; }

  void describeHazard(llvm::raw_ostream &Os, const Expr *Origin,
                      QualType SinkType) const override {
    QualType SinkObject = pointeeType(SinkType);
    if (!SinkObject.isNull() && isBorrowType(SinkObject)) {
      Os << "Borrow that does not travel with the lambda";
      return;
    }

    Os << "loan on ";
    QualType OriginType = Origin ? pointeeType(Origin->getType()) : QualType();

    // Name the borrowed type, not the Borrow<T> guard, when the loan was
    // taken from a Borrow<T>.
    if (!OriginType.isNull() && isBorrowType(OriginType))
      OriginType = borrowedType(OriginType);

    if (!OriginType.isNull() && OriginType->getAsRecordDecl()) {
      Os << "CanBorrow type ";
      printTypeName(Os, OriginType);
    } else
      Os << "a CanBorrow object";
    Os << " that is not guarded by const or a Borrow";
  }
};

} // namespace

std::optional<bool> clang::isUnsafePtrForStorage(const PtrRefSafetyModel &Model,
                                                 QualType T, bool IgnoreARC) {
  // A __strong / __weak Objective-C storage location is memory managed and
  // thus safe. This exemption applies to variables/members/captures but not to
  // call arguments, so it lives here rather than in the policy itself.
  if (Model.retainTypeChecker() && T.hasStrongOrWeakObjCLifetime())
    return false;
  return Model.isUnsafePtr(T, IgnoreARC);
}

std::unique_ptr<PtrRefSafetyModel> clang::makeRefPtrSafetyModel() {
  return std::make_unique<RefCountedSafetyModel>();
}

std::unique_ptr<PtrRefSafetyModel> clang::makeCheckedPtrSafetyModel() {
  return std::make_unique<CheckedPtrSafetyModel>();
}

std::unique_ptr<PtrRefSafetyModel> clang::makeRetainPtrSafetyModel() {
  return std::make_unique<RetainPtrSafetyModel>();
}

std::unique_ptr<PtrRefSafetyModel> clang::makeBorrowSafetyModel() {
  return std::make_unique<BorrowSafetyModel>();
}
