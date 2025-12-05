//=======- PtrTypesSemantics.cpp ---------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYZER_WEBKIT_PTRTYPESEMANTICS_H
#define LLVM_CLANG_ANALYZER_WEBKIT_PTRTYPESEMANTICS_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include <optional>

namespace clang {
class CXXBaseSpecifier;
class CXXMethodDecl;
class CXXRecordDecl;
class Decl;
class FunctionDecl;
class NamedDecl;
class QualType;
class RecordType;
class Stmt;
class TranslationUnitDecl;
class Type;
class TypedefDecl;

// Ref-countability of a type is implicitly defined by Ref<T> and RefPtr<T>
// implementation. It can be modeled as: type T having public methods ref() and
// deref()

// In WebKit there are two ref-counted templated smart pointers: RefPtr<T> and
// Ref<T>.

/// \returns CXXRecordDecl of the base if the type has ref as a public method,
/// nullptr if not, std::nullopt if inconclusive.
std::optional<const clang::CXXRecordDecl *>
hasPublicMethodInBase(const CXXBaseSpecifier *Base,
                      llvm::StringRef NameToMatch);

/// \returns true if \p Class is ref-countable, false if not, std::nullopt if
/// inconclusive.
std::optional<bool> isRefCountable(const clang::CXXRecordDecl *Class);

/// \returns true if \p Class is checked-pointer compatible, false if not,
/// std::nullopt if inconclusive.
std::optional<bool> isCheckedPtrCapable(const clang::CXXRecordDecl *Class);

/// \returns true if \p Class is ref-counted, false if not.
bool isRefCounted(const clang::CXXRecordDecl *Class);

/// \returns true if \p Class is a CheckedPtr / CheckedRef, false if not.
bool isCheckedPtr(const clang::CXXRecordDecl *Class);

/// \returns true if \p Class is a RetainPtr, false if not.
bool isRetainPtrOrOSPtr(const clang::CXXRecordDecl *Class);

/// \returns true if \p Class is a smart pointer (RefPtr, WeakPtr, etc...),
/// false if not.
bool isSmartPtr(const clang::CXXRecordDecl *Class);

/// \returns true if \p Class is ref-countable AND not ref-counted, false if
/// not, std::nullopt if inconclusive.
std::optional<bool> isUncounted(const clang::QualType T);

/// \returns true if \p Class is CheckedPtr capable AND not checked, false if
/// not, std::nullopt if inconclusive.
std::optional<bool> isUnchecked(const clang::QualType T);

/// An inter-procedural analysis facility that detects CF types with the
/// underlying pointer type.
class RetainTypeChecker {
  llvm::DenseSet<const RecordType *> CFPointees;
  llvm::DenseSet<const Type *> RecordlessTypes;
  bool IsARCEnabled{false};
  bool DefaultSynthProperties{true};

public:
  void visitTranslationUnitDecl(const TranslationUnitDecl *);
  void visitTypedef(const TypedefDecl *);
  bool isUnretained(const QualType, bool ignoreARC = false);
  bool isARCEnabled() const { return IsARCEnabled; }
  bool defaultSynthProperties() const { return DefaultSynthProperties; }
};

/// \returns true if \p Class is NS or CF objects AND not retained, false if
/// not, std::nullopt if inconclusive.
std::optional<bool> isUnretained(const clang::QualType T, bool IsARCEnabled);

/// \returns true if \p Class is ref-countable AND not ref-counted, false if
/// not, std::nullopt if inconclusive.
std::optional<bool> isUncounted(const clang::CXXRecordDecl* Class);

/// \returns true if \p Class is CheckedPtr capable AND not checked, false if
/// not, std::nullopt if inconclusive.
std::optional<bool> isUnchecked(const clang::CXXRecordDecl *Class);

/// \returns true if \p T is either a raw pointer or reference to an uncounted
/// class, false if not, std::nullopt if inconclusive.
std::optional<bool> isUncountedPtr(const clang::QualType T);

/// \returns true if \p T is either a raw pointer or reference to an unchecked
/// class, false if not, std::nullopt if inconclusive.
std::optional<bool> isUncheckedPtr(const clang::QualType T);

/// \returns true if \p T is either a raw pointer or reference to an uncounted
/// or unchecked class, false if not, std::nullopt if inconclusive.
std::optional<bool> isUnsafePtr(const QualType T, bool IsArcEnabled);

/// \returns true if \p T is a RefPtr, Ref, CheckedPtr, CheckedRef, or its
/// variant, false if not.
bool isRefOrCheckedPtrType(const clang::QualType T);

/// \returns true if \p T is a RetainPtr, false if not.
bool isRetainPtrOrOSPtrType(const clang::QualType T);

/// \returns true if \p T is a RefPtr, Ref, CheckedPtr, CheckedRef, or
/// unique_ptr, false if not.
bool isOwnerPtrType(const clang::QualType T);

/// \returns true if \p F creates ref-countable object from uncounted parameter,
/// false if not.
bool isCtorOfRefCounted(const clang::FunctionDecl *F);

/// \returns true if \p F creates checked ptr object from uncounted parameter,
/// false if not.
bool isCtorOfCheckedPtr(const clang::FunctionDecl *F);

/// \returns true if \p F creates ref-countable or checked ptr object from
/// uncounted parameter, false if not.
bool isCtorOfSafePtr(const clang::FunctionDecl *F);

/// \returns true if \p Name is RefPtr, Ref, or its variant, false if not.
bool isRefType(const std::string &Name);

/// \returns true if \p Name is CheckedRef or CheckedPtr, false if not.
bool isCheckedPtr(const std::string &Name);

/// \returns true if \p Name is RetainPtr or its variant, false if not.
bool isRetainPtrOrOSPtr(const std::string &Name);

/// \returns true if \p Name is an owning smar pointer such as Ref, CheckedPtr,
/// and unique_ptr.
bool isOwnerPtr(const std::string &Name);

/// \returns true if \p Name is a smart pointer type name, false if not.
bool isSmartPtrClass(const std::string &Name);

/// \returns true if \p M is getter of a ref-counted class, false if not.
std::optional<bool> isGetterOfSafePtr(const clang::CXXMethodDecl *Method);

/// \returns true if \p F is a conversion between ref-countable or ref-counted
/// pointer types.
bool isPtrConversion(const FunctionDecl *F);

/// \returns true if \p F is a builtin function which is considered trivial.
bool isTrivialBuiltinFunction(const FunctionDecl *F);

/// \returns true if \p F is a static singleton function.
bool isSingleton(const NamedDecl *F);

/// An inter-procedural analysis facility that detects functions with "trivial"
/// behavior with respect to reference counting, such as simple field getters.
class TrivialFunctionAnalysis {
public:
  /// \returns true if \p D is a "trivial" function.
  bool isTrivial(const Decl *D) const { return isTrivialImpl(D, TheCache); }
  bool isTrivial(const Stmt *S) const { return isTrivialImpl(S, TheCache); }

private:
  friend class TrivialFunctionAnalysisVisitor;

  using CacheTy =
      llvm::DenseMap<llvm::PointerUnion<const Decl *, const Stmt *>, bool>;
  mutable CacheTy TheCache{};

  static bool isTrivialImpl(const Decl *D, CacheTy &Cache);
  static bool isTrivialImpl(const Stmt *S, CacheTy &Cache);
};

} // namespace clang

#endif
