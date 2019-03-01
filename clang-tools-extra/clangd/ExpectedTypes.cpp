//===--- ExpectedTypes.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpectedTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace clangd {
namespace {

static const Type *toEquivClass(ASTContext &Ctx, QualType T) {
  if (T.isNull() || T->isDependentType())
    return nullptr;
  // Drop references, we do not handle reference inits properly anyway.
  T = T.getCanonicalType().getNonReferenceType();
  // Numeric types are the simplest case.
  if (T->isBooleanType())
    return Ctx.BoolTy.getTypePtr();
  if (T->isIntegerType() && !T->isEnumeralType())
    return Ctx.IntTy.getTypePtr(); // All integers are equivalent.
  if (T->isFloatingType() && !T->isComplexType())
    return Ctx.FloatTy.getTypePtr(); // All floats are equivalent.

  // Do some simple transformations.
  if (T->isArrayType()) // Decay arrays to pointers.
    return Ctx.getPointerType(QualType(T->getArrayElementTypeNoTypeQual(), 0))
        .getTypePtr();
  // Drop the qualifiers and return the resulting type.
  // FIXME: also drop qualifiers from pointer types, e.g. 'const T* => T*'
  return T.getTypePtr();
}

static llvm::Optional<QualType>
typeOfCompletion(const CodeCompletionResult &R) {
  auto *VD = dyn_cast_or_null<ValueDecl>(R.Declaration);
  if (!VD)
    return llvm::None; // We handle only variables and functions below.
  auto T = VD->getType();
  if (T.isNull())
    return llvm::None;
  if (auto FuncT = T->getAs<FunctionType>()) {
    // Functions are a special case. They are completed as 'foo()' and we want
    // to match their return type rather than the function type itself.
    // FIXME(ibiryukov): in some cases, we might want to avoid completing `()`
    // after the function name, e.g. `std::cout << std::endl`.
    return FuncT->getReturnType();
  }
  return T;
}
} // namespace

llvm::Optional<OpaqueType> OpaqueType::encode(ASTContext &Ctx, QualType T) {
  if (T.isNull())
    return None;
  const Type *C = toEquivClass(Ctx, T);
  if (!C)
    return None;
  llvm::SmallString<128> Encoded;
  if (index::generateUSRForType(QualType(C, 0), Ctx, Encoded))
    return None;
  return OpaqueType(Encoded.str());
}

OpaqueType::OpaqueType(std::string Data) : Data(std::move(Data)) {}

llvm::Optional<OpaqueType> OpaqueType::fromType(ASTContext &Ctx,
                                                QualType Type) {
  return encode(Ctx, Type);
}

llvm::Optional<OpaqueType>
OpaqueType::fromCompletionResult(ASTContext &Ctx,
                                 const CodeCompletionResult &R) {
  auto T = typeOfCompletion(R);
  if (!T)
    return None;
  return encode(Ctx, *T);
}

} // namespace clangd
} // namespace clang
