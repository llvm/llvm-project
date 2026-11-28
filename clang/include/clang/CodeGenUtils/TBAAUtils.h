//===--- TBAAUtils.h - Shared TBAA queries ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the AST queries that both classic CodeGen and CIR CodeGen
// need while computing type-based alias analysis (TBAA) information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_TBAAUTILS_H
#define LLVM_CLANG_CODEGENUTILS_TBAAUTILS_H

#include "clang/AST/Type.h"

namespace clang::CodeGenUtils {

/// Returns true if accesses to objects of type \p QTy are in the general char
/// alias class because of a may_alias attribute.
bool typeHasMayAlias(QualType QTy);

/// Returns true if \p QTy is a valid base type to be used in TBAA access tags.
bool isValidTBAABaseType(QualType QTy);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_TBAAUTILS_H
