//===- ASTCommon.h - Common stuff for ASTReader/ASTWriter -*- C++ -*-=========//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines common functions that both ASTReader and ASTWriter use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_LIB_AST_COMMON_H
#define LLVM_CLANG_SERIALIZATION_LIB_AST_COMMON_H

#include "clang/Serialization/ASTBitCodes.h"
#include "clang/AST/ASTContext.h"

namespace clang {

namespace serialization {

enum DeclUpdateKind {
  UPD_CXX_ADDED_IMPLICIT_MEMBER,
  UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION,
  UPD_CXX_ADDED_ANONYMOUS_NAMESPACE,
  UPD_CXX_INSTANTIATED_STATIC_DATA_MEMBER
};

TypeIdx TypeIdxFromBuiltin(const BuiltinType *BT);

template <typename IdxForTypeTy>
TypeID MakeTypeID(ASTContext &Context, QualType T, IdxForTypeTy IdxForType) {
  if (T.isNull())
    return PREDEF_TYPE_NULL_ID;

  unsigned FastQuals = T.getLocalFastQualifiers();
  T.removeLocalFastQualifiers();

  if (T.hasLocalNonFastQualifiers())
    return IdxForType(T).asTypeID(FastQuals);

  assert(!T.hasLocalQualifiers());

  if (const BuiltinType *BT = dyn_cast<BuiltinType>(T.getTypePtr()))
    return TypeIdxFromBuiltin(BT).asTypeID(FastQuals);

  if (T == Context.AutoDeductTy)
    return TypeIdx(PREDEF_TYPE_AUTO_DEDUCT).asTypeID(FastQuals);
  if (T == Context.AutoRRefDeductTy)
    return TypeIdx(PREDEF_TYPE_AUTO_RREF_DEDUCT).asTypeID(FastQuals);
  if (T == Context.VaListTagTy)
    return TypeIdx(PREDEF_TYPE_VA_LIST_TAG).asTypeID(FastQuals);

  return IdxForType(T).asTypeID(FastQuals);
}

unsigned ComputeHash(Selector Sel);

} // namespace serialization

} // namespace clang

#endif
