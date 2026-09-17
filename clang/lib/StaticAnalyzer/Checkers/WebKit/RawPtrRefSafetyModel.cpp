//=======- RawPtrRefSafetyModel.cpp -----------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RawPtrRefSafetyModel.h"
#include "ASTUtils.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"

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
  bool isSafeExpr(const Expr *E) const override {
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
  bool isSafeDecl(const Decl *D, const SourceManager &SM) const override {
    // Treat NS/CF globals in system header as immortal.
    return SM.isInSystemHeader(D->getLocation());
  }
  const char *typeName() const override { return "RetainPtr-capable type"; }
  RetainTypeChecker *retainTypeChecker() const override { return &RTC; }
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
