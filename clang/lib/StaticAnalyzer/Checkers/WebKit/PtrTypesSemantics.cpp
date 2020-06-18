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

using llvm::Optional;
using namespace clang;

namespace {

bool hasPublicRefAndDeref(const CXXRecordDecl *R) {
  assert(R);

  bool hasRef = false;
  bool hasDeref = false;
  for (const CXXMethodDecl *MD : R->methods()) {
    const auto MethodName = safeGetName(MD);

    if (MethodName == "ref" && MD->getAccess() == AS_public) {
      if (hasDeref)
        return true;
      hasRef = true;
    } else if (MethodName == "deref" && MD->getAccess() == AS_public) {
      if (hasRef)
        return true;
      hasDeref = true;
    }
  }
  return false;
}

} // namespace

namespace clang {

const CXXRecordDecl *isRefCountable(const CXXBaseSpecifier *Base) {
  assert(Base);

  const Type *T = Base->getType().getTypePtrOrNull();
  if (!T)
    return nullptr;

  const CXXRecordDecl *R = T->getAsCXXRecordDecl();
  if (!R)
    return nullptr;

  return hasPublicRefAndDeref(R) ? R : nullptr;
}

bool isRefCountable(const CXXRecordDecl *R) {
  assert(R);

  R = R->getDefinition();
  assert(R);

  if (hasPublicRefAndDeref(R))
    return true;

  CXXBasePaths Paths;
  Paths.setOrigin(const_cast<CXXRecordDecl *>(R));

  const auto isRefCountableBase = [](const CXXBaseSpecifier *Base,
                                     CXXBasePath &) {
    return clang::isRefCountable(Base);
  };

  return R->lookupInBases(isRefCountableBase, Paths,
                          /*LookupInDependent =*/true);
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

bool isUncounted(const CXXRecordDecl *Class) {
  // Keep isRefCounted first as it's cheaper.
  return !isRefCounted(Class) && isRefCountable(Class);
}

bool isUncountedPtr(const Type *T) {
  assert(T);

  if (T->isPointerType() || T->isReferenceType()) {
    if (auto *CXXRD = T->getPointeeCXXRecordDecl()) {
      return isUncounted(CXXRD);
    }
  }
  return false;
}

bool isGetterOfRefCounted(const CXXMethodDecl *M) {
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
          if (isUncountedPtr(targetConversionType)) {
            return true;
          }
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
      FunctionName == "makeWeakPtr"

      || FunctionName == "downcast" || FunctionName == "bitwise_cast")
    return true;

  return false;
}

} // namespace clang
