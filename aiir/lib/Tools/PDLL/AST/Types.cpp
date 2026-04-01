//===- Types.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Tools/PDLL/AST/Types.h"
#include "aiir/Tools/PDLL/AST/Context.h"
#include <optional>

using namespace aiir;
using namespace aiir::pdll;
using namespace aiir::pdll::ast;

AIIR_DEFINE_EXPLICIT_TYPE_ID(aiir::pdll::ast::detail::AttributeTypeStorage)
AIIR_DEFINE_EXPLICIT_TYPE_ID(aiir::pdll::ast::detail::ConstraintTypeStorage)
AIIR_DEFINE_EXPLICIT_TYPE_ID(aiir::pdll::ast::detail::OperationTypeStorage)
AIIR_DEFINE_EXPLICIT_TYPE_ID(aiir::pdll::ast::detail::RangeTypeStorage)
AIIR_DEFINE_EXPLICIT_TYPE_ID(aiir::pdll::ast::detail::RewriteTypeStorage)
AIIR_DEFINE_EXPLICIT_TYPE_ID(aiir::pdll::ast::detail::TupleTypeStorage)
AIIR_DEFINE_EXPLICIT_TYPE_ID(aiir::pdll::ast::detail::TypeTypeStorage)
AIIR_DEFINE_EXPLICIT_TYPE_ID(aiir::pdll::ast::detail::ValueTypeStorage)

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

TypeID Type::getTypeID() const { return impl->typeID; }

Type Type::refineWith(Type other) const {
  if (*this == other)
    return *this;

  // Operation types are compatible if the operation names don't conflict.
  if (auto opTy = aiir::dyn_cast<OperationType>(*this)) {
    auto otherOpTy = aiir::dyn_cast<ast::OperationType>(other);
    if (!otherOpTy)
      return nullptr;
    if (!otherOpTy.getName())
      return *this;
    if (!opTy.getName())
      return other;

    return nullptr;
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// AttributeType
//===----------------------------------------------------------------------===//

AttributeType AttributeType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}

//===----------------------------------------------------------------------===//
// ConstraintType
//===----------------------------------------------------------------------===//

ConstraintType ConstraintType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}

//===----------------------------------------------------------------------===//
// OperationType
//===----------------------------------------------------------------------===//

OperationType OperationType::get(Context &context,
                                 std::optional<StringRef> name,
                                 const ods::Operation *odsOp) {
  return context.getTypeUniquer().get<ImplTy>(
      /*initFn=*/function_ref<void(ImplTy *)>(),
      std::make_pair(name.value_or(""), odsOp));
}

std::optional<StringRef> OperationType::getName() const {
  StringRef name = getImplAs<ImplTy>()->getValue().first;
  return name.empty() ? std::optional<StringRef>()
                      : std::optional<StringRef>(name);
}

const ods::Operation *OperationType::getODSOperation() const {
  return getImplAs<ImplTy>()->getValue().second;
}

//===----------------------------------------------------------------------===//
// RangeType
//===----------------------------------------------------------------------===//

RangeType RangeType::get(Context &context, Type elementType) {
  return context.getTypeUniquer().get<ImplTy>(
      /*initFn=*/function_ref<void(ImplTy *)>(), elementType);
}

Type RangeType::getElementType() const {
  return getImplAs<ImplTy>()->getValue();
}

//===----------------------------------------------------------------------===//
// TypeRangeType
//===----------------------------------------------------------------------===//

bool TypeRangeType::classof(Type type) {
  RangeType range = aiir::dyn_cast<RangeType>(type);
  return range && aiir::isa<TypeType>(range.getElementType());
}

TypeRangeType TypeRangeType::get(Context &context) {
  return aiir::cast<TypeRangeType>(
      RangeType::get(context, TypeType::get(context)));
}

//===----------------------------------------------------------------------===//
// ValueRangeType
//===----------------------------------------------------------------------===//

bool ValueRangeType::classof(Type type) {
  RangeType range = aiir::dyn_cast<RangeType>(type);
  return range && aiir::isa<ValueType>(range.getElementType());
}

ValueRangeType ValueRangeType::get(Context &context) {
  return aiir::cast<ValueRangeType>(
      RangeType::get(context, ValueType::get(context)));
}

//===----------------------------------------------------------------------===//
// RewriteType
//===----------------------------------------------------------------------===//

RewriteType RewriteType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//

TupleType TupleType::get(Context &context, ArrayRef<Type> elementTypes,
                         ArrayRef<StringRef> elementNames) {
  assert(elementTypes.size() == elementNames.size());
  return context.getTypeUniquer().get<ImplTy>(
      /*initFn=*/function_ref<void(ImplTy *)>(), elementTypes, elementNames);
}
TupleType TupleType::get(Context &context, ArrayRef<Type> elementTypes) {
  SmallVector<StringRef> elementNames(elementTypes.size());
  return get(context, elementTypes, elementNames);
}

ArrayRef<Type> TupleType::getElementTypes() const {
  return getImplAs<ImplTy>()->getValue().first;
}

ArrayRef<StringRef> TupleType::getElementNames() const {
  return getImplAs<ImplTy>()->getValue().second;
}

//===----------------------------------------------------------------------===//
// TypeType
//===----------------------------------------------------------------------===//

TypeType TypeType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}

//===----------------------------------------------------------------------===//
// ValueType
//===----------------------------------------------------------------------===//

ValueType ValueType::get(Context &context) {
  return context.getTypeUniquer().get<ImplTy>();
}
