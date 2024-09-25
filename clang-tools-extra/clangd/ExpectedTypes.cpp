//===--- ExpectedTypes.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExpectedTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Type.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include <optional>

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

static std::optional<QualType> typeOfCompletion(const CodeCompletionResult &R) {
  const NamedDecl *D = R.Declaration;
  // Templates do not have a type on their own, look at the templated decl.
  if (auto *Template = dyn_cast_or_null<TemplateDecl>(D))
    D = Template->getTemplatedDecl();
  auto *VD = dyn_cast_or_null<ValueDecl>(D);
  if (!VD)
    return std::nullopt; // We handle only variables and functions below.
  auto T = VD->getType();
  if (T.isNull())
    return std::nullopt;
  if (auto *FuncT = T->getAs<FunctionType>()) {
    // Functions are a special case. They are completed as 'foo()' and we want
    // to match their return type rather than the function type itself.
    // FIXME(ibiryukov): in some cases, we might want to avoid completing `()`
    // after the function name, e.g. `std::cout << std::endl`.
    return FuncT->getReturnType();
  }
  return T;
}
} // namespace

std::optional<OpaqueType> OpaqueType::encode(ASTContext &Ctx, QualType T) {
  if (T.isNull())
    return std::nullopt;
  const Type *C = toEquivClass(Ctx, T);
  if (!C)
    return std::nullopt;
  llvm::SmallString<128> Encoded;
  if (index::generateUSRForType(QualType(C, 0), Ctx, Encoded))
    return std::nullopt;
  return OpaqueType(std::string(Encoded));
}

OpaqueType::OpaqueType(std::string Data) : Data(std::move(Data)) {}

std::optional<OpaqueType> OpaqueType::fromType(ASTContext &Ctx, QualType Type) {
  return encode(Ctx, Type);
}

std::optional<OpaqueType>
OpaqueType::fromCompletionResult(ASTContext &Ctx,
                                 const CodeCompletionResult &R) {
  auto T = typeOfCompletion(R);
  if (!T)
    return std::nullopt;
  return encode(Ctx, *T);
}

} // namespace clangd
} // namespace clang
