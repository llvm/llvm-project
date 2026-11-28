//===--- TBAAUtils.cpp - Shared TBAA queries ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/TBAAUtils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"

namespace clang::CodeGenUtils {

bool typeHasMayAlias(QualType QTy) {
  // Tagged types have declarations, and therefore may have attributes.
  if (auto *TD = QTy->getAsTagDecl())
    if (TD->hasAttr<MayAliasAttr>())
      return true;

  // Also look for may_alias as a declaration attribute on a typedef.
  // FIXME: We should follow GCC and model may_alias as a type attribute
  // rather than as a declaration attribute.
  while (auto *TT = QTy->getAs<TypedefType>()) {
    if (TT->getDecl()->hasAttr<MayAliasAttr>())
      return true;
    QTy = TT->desugar();
  }

  // Also consider an array type as may_alias when its element type (at
  // any level) is marked as such.
  if (auto *ArrayTy = QTy->getAsArrayTypeUnsafe())
    if (typeHasMayAlias(ArrayTy->getElementType()))
      return true;

  return false;
}

bool isValidTBAABaseType(QualType QTy) {
  if (const auto *RD = QTy->getAsRecordDecl()) {
    // Incomplete types are not valid base access types.
    if (!RD->isCompleteDefinition())
      return false;
    if (RD->hasFlexibleArrayMember())
      return false;
    // RD can be struct, union, class, interface or enum.
    // For now, we only handle struct and class.
    if (RD->isStruct() || RD->isClass())
      return true;
  }
  return false;
}

} // namespace clang::CodeGenUtils
