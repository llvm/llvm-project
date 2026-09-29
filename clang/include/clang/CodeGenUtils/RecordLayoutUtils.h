//===--- RecordLayoutUtils.h - Shared record layout queries -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the AST queries about record layout that both classic
// CodeGen and CIR CodeGen need while lowering a record to its target type.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_RECORDLAYOUTUTILS_H
#define LLVM_CLANG_CODEGENUTILS_RECORDLAYOUTUTILS_H

#include "clang/AST/ASTContext.h"

namespace clang::CodeGenUtils {

/// Return true iff the field is "empty", that is, either a zero-width
/// bit-field or an \ref isEmptyRecordForLayout.
bool isEmptyFieldForLayout(const ASTContext &Ctx, const FieldDecl *FD);

/// Return true iff a structure contains only empty base classes (per \ref
/// isEmptyRecordForLayout) and fields (per \ref isEmptyFieldForLayout).  Note,
/// C++ record fields are considered empty if the [[no_unique_address]]
/// attribute would have made them empty, so this is not the same as \ref
/// isEmptyRecord.
bool isEmptyRecordForLayout(const ASTContext &Ctx, QualType T);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_RECORDLAYOUTUTILS_H
