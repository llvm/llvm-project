//=======- RawPtrRefSafetyModel.h -------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYZER_WEBKIT_RAWPTRREFSAFETYMODEL_H
#define LLVM_CLANG_ANALYZER_WEBKIT_RAWPTRREFSAFETYMODEL_H

#include "PtrTypesSemantics.h"
#include <memory>
#include <optional>
#include <string>

namespace clang {
class CXXRecordDecl;
class Decl;
class Expr;
class QualType;
class SourceManager;

/// Models one WebKit pointer-safety policy: ref-counted (RefPtr), checked
/// (CheckedPtr), or retainable (RetainPtr/OSPtr).
///
/// It captures the family-specific "what is a safe/unsafe pointer" questions
/// that are shared by the various RawPtrRef* checkers, independently of how
/// each checker traverses the AST (call arguments, local variables, members,
/// lambda captures, ...). This lets a single policy be defined once and reused
/// across every traversal.
class PtrRefSafetyModel {
public:
  virtual ~PtrRefSafetyModel() = default;

  /// \returns whether \p QT itself is an unsafe (smart-pointer-capable but not
  /// managed) type, false if not, std::nullopt if inconclusive.
  virtual std::optional<bool> isUnsafeType(QualType QT) const = 0;

  /// \returns whether \p QT is a raw pointer or reference to an unsafe type,
  /// false if not, std::nullopt if inconclusive. \p IgnoreARC requests that
  /// Objective-C ARC be ignored when deciding retainability.
  virtual std::optional<bool> isUnsafePtr(QualType QT,
                                          bool IgnoreARC = false) const = 0;

  /// \returns whether \p Record is a safe smart pointer for this policy.
  virtual bool isSafePtr(const CXXRecordDecl *Record) const = 0;

  /// \returns whether \p T is a safe smart pointer type for this policy.
  virtual bool isSafePtrType(QualType T) const = 0;

  /// \returns whether \p Name is the name of a safe smart pointer class for
  /// this policy.
  virtual bool isPtrType(const std::string &Name) const = 0;

  /// \returns whether \p E is known to produce a safe value for this policy.
  virtual bool isSafeExpr(const Expr *) const { return false; }

  /// \returns whether \p D refers to a declaration that is safe by construction
  /// for this policy (e.g. immortal system-header globals).
  virtual bool isSafeDecl(const Decl *, const SourceManager &) const {
    return false;
  }

  /// \returns a human readable name for the safe type category, used in
  /// diagnostics (e.g. "RefPtr-capable type").
  virtual const char *typeName() const = 0;

  /// \returns the RetainTypeChecker backing this policy, or nullptr if the
  /// policy does not track retain/OS types.
  virtual RetainTypeChecker *retainTypeChecker() const { return nullptr; }
};

/// Applies the memory-management exemptions that hold for a variable, member,
/// or lambda capture (but not for a call argument) before consulting \p Model:
/// a __strong / __weak Objective-C storage location is memory managed and thus
/// safe. \returns whether \p T is an unsafe pointer in such a storage context.
std::optional<bool> isUnsafePtrForStorage(const PtrRefSafetyModel &Model,
                                          QualType T, bool IgnoreARC = false);

/// \returns a policy that treats ref-counted / checked pointers as safe.
std::unique_ptr<PtrRefSafetyModel> makeRefPtrSafetyModel();

/// \returns a policy that treats checked pointers as safe.
std::unique_ptr<PtrRefSafetyModel> makeCheckedPtrSafetyModel();

/// \returns a policy that treats RetainPtr / OSPtr as safe.
std::unique_ptr<PtrRefSafetyModel> makeRetainPtrSafetyModel();

} // namespace clang

#endif
