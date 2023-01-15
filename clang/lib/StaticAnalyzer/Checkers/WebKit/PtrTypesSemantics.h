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
#include <optional>

namespace clang {
class CXXBaseSpecifier;
class CXXMethodDecl;
class CXXRecordDecl;
class FunctionDecl;
class Type;

// Ref-countability of a type is implicitly defined by Ref<T> and RefPtr<T>
// implementation. It can be modeled as: type T having public methods ref() and
// deref()

// In WebKit there are two ref-counted templated smart pointers: RefPtr<T> and
// Ref<T>.

/// \returns CXXRecordDecl of the base if the type is ref-countable, nullptr if
/// not, std::nullopt if inconclusive.
std::optional<const clang::CXXRecordDecl*>
isRefCountable(const clang::CXXBaseSpecifier* Base);

/// \returns true if \p Class is ref-countable, false if not, std::nullopt if
/// inconclusive.
std::optional<bool> isRefCountable(const clang::CXXRecordDecl* Class);

/// \returns true if \p Class is ref-counted, false if not.
bool isRefCounted(const clang::CXXRecordDecl *Class);

/// \returns true if \p Class is ref-countable AND not ref-counted, false if
/// not, std::nullopt if inconclusive.
std::optional<bool> isUncounted(const clang::CXXRecordDecl* Class);

/// \returns true if \p T is either a raw pointer or reference to an uncounted
/// class, false if not, std::nullopt if inconclusive.
std::optional<bool> isUncountedPtr(const clang::Type* T);

/// \returns true if \p F creates ref-countable object from uncounted parameter,
/// false if not.
bool isCtorOfRefCounted(const clang::FunctionDecl *F);

/// \returns true if \p M is getter of a ref-counted class, false if not.
std::optional<bool> isGetterOfRefCounted(const clang::CXXMethodDecl* Method);

/// \returns true if \p F is a conversion between ref-countable or ref-counted
/// pointer types.
bool isPtrConversion(const FunctionDecl *F);

} // namespace clang

#endif
