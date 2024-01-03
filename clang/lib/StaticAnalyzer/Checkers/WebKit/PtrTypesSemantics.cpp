//=======- PtrTypesSemantics.cpp ---------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PtrTypesSemantics.h"
#include "ASTUtils.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include <optional>

using namespace clang;

namespace {

bool hasPublicMethodInBaseClass(const CXXRecordDecl *R,
                                const char *NameToMatch) {
  assert(R);
  assert(R->hasDefinition());

  for (const CXXMethodDecl *MD : R->methods()) {
    const auto MethodName = safeGetName(MD);
    if (MethodName == NameToMatch && MD->getAccess() == AS_public)
      return true;
  }
  return false;
}

} // namespace

namespace clang {

std::optional<const clang::CXXRecordDecl *>
hasPublicMethodInBase(const CXXBaseSpecifier *Base, const char *NameToMatch) {
  assert(Base);

  const Type *T = Base->getType().getTypePtrOrNull();
  if (!T)
    return std::nullopt;

  const CXXRecordDecl *R = T->getAsCXXRecordDecl();
  if (!R)
    return std::nullopt;
  if (!R->hasDefinition())
    return std::nullopt;

  return hasPublicMethodInBaseClass(R, NameToMatch) ? R : nullptr;
}

std::optional<bool> isRefCountable(const CXXRecordDecl* R)
{
  assert(R);

  R = R->getDefinition();
  if (!R)
    return std::nullopt;

  bool hasRef = hasPublicMethodInBaseClass(R, "ref");
  bool hasDeref = hasPublicMethodInBaseClass(R, "deref");
  if (hasRef && hasDeref)
    return true;

  CXXBasePaths Paths;
  Paths.setOrigin(const_cast<CXXRecordDecl *>(R));

  bool AnyInconclusiveBase = false;
  const auto hasPublicRefInBase =
      [&AnyInconclusiveBase](const CXXBaseSpecifier *Base, CXXBasePath &) {
        auto hasRefInBase = clang::hasPublicMethodInBase(Base, "ref");
        if (!hasRefInBase) {
          AnyInconclusiveBase = true;
          return false;
        }
        return (*hasRefInBase) != nullptr;
      };

  hasRef = hasRef || R->lookupInBases(hasPublicRefInBase, Paths,
                                      /*LookupInDependent =*/true);
  if (AnyInconclusiveBase)
    return std::nullopt;

  const auto hasPublicDerefInBase =
      [&AnyInconclusiveBase](const CXXBaseSpecifier *Base, CXXBasePath &) {
        auto hasDerefInBase = clang::hasPublicMethodInBase(Base, "deref");
        if (!hasDerefInBase) {
          AnyInconclusiveBase = true;
          return false;
        }
        return (*hasDerefInBase) != nullptr;
      };
  hasDeref = hasDeref || R->lookupInBases(hasPublicDerefInBase, Paths,
                                          /*LookupInDependent =*/true);
  if (AnyInconclusiveBase)
    return std::nullopt;

  return hasRef && hasDeref;
}

bool isCtorOfRefCounted(const clang::FunctionDecl *F) {
  assert(F);
  const auto &FunctionName = safeGetName(F);

  return FunctionName == "Ref" || FunctionName == "makeRef"

         || FunctionName == "RefPtr" || FunctionName == "makeRefPtr"

         || FunctionName == "UniqueRef" || FunctionName == "makeUniqueRef" ||
         FunctionName == "makeUniqueRefWithoutFastMallocCheck"

         || FunctionName == "String" || FunctionName == "AtomString" ||
         FunctionName == "UniqueString"
         // FIXME: Implement as attribute.
         || FunctionName == "Identifier";
}

std::optional<bool> isUncounted(const CXXRecordDecl* Class)
{
  // Keep isRefCounted first as it's cheaper.
  if (isRefCounted(Class))
    return false;

  std::optional<bool> IsRefCountable = isRefCountable(Class);
  if (!IsRefCountable)
    return std::nullopt;

  return (*IsRefCountable);
}

std::optional<bool> isUncountedPtr(const Type* T)
{
  assert(T);

  if (T->isPointerType() || T->isReferenceType()) {
    if (auto *CXXRD = T->getPointeeCXXRecordDecl()) {
      return isUncounted(CXXRD);
    }
  }
  return false;
}

std::optional<bool> isGetterOfRefCounted(const CXXMethodDecl* M)
{
  assert(M);

  if (isa<CXXMethodDecl>(M)) {
    const CXXRecordDecl *calleeMethodsClass = M->getParent();
    auto className = safeGetName(calleeMethodsClass);
    auto methodName = safeGetName(M);

    if (((className == "Ref" || className == "RefPtr") &&
         methodName == "get") ||
        ((className == "String" || className == "AtomString" ||
          className == "AtomStringImpl" || className == "UniqueString" ||
          className == "UniqueStringImpl" || className == "Identifier") &&
         methodName == "impl"))
      return true;

    // Ref<T> -> T conversion
    // FIXME: Currently allowing any Ref<T> -> whatever cast.
    if (className == "Ref" || className == "RefPtr") {
      if (auto *maybeRefToRawOperator = dyn_cast<CXXConversionDecl>(M)) {
        if (auto *targetConversionType =
                maybeRefToRawOperator->getConversionType().getTypePtrOrNull()) {
          return isUncountedPtr(targetConversionType);
        }
      }
    }
  }
  return false;
}

bool isRefCounted(const CXXRecordDecl *R) {
  assert(R);
  if (auto *TmplR = R->getTemplateInstantiationPattern()) {
    // FIXME: String/AtomString/UniqueString
    const auto &ClassName = safeGetName(TmplR);
    return ClassName == "RefPtr" || ClassName == "Ref";
  }
  return false;
}

bool isPtrConversion(const FunctionDecl *F) {
  assert(F);
  if (isCtorOfRefCounted(F))
    return true;

  // FIXME: check # of params == 1
  const auto FunctionName = safeGetName(F);
  if (FunctionName == "getPtr" || FunctionName == "WeakPtr" ||
      FunctionName == "dynamicDowncast"
      || FunctionName == "downcast" || FunctionName == "bitwise_cast")
    return true;

  return false;
}

} // namespace clang
