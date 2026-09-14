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

/// Recursively searches all of the bases of \p Decl to find out whether
/// \p Query is not the primary vbase of some base class.
bool hasOwnStorage(const ASTContext &Ctx, const CXXRecordDecl *Decl,
                   const CXXRecordDecl *Query);

/// The Microsoft bitfield layout rule allocates discrete storage units of the
/// field's formal type and only combines adjacent fields of the same formal
/// type.  We want to emit a layout with these discrete storage units instead
/// of combining them into a continuous run.
bool isDiscreteBitFieldABI(const ASTContext &Ctx, const RecordDecl *RD);

/// Return true iff the field is "empty", that is, either a zero-width
/// bit-field or an \ref isEmptyRecordForLayout.
bool isEmptyFieldForLayout(const ASTContext &Ctx, const FieldDecl *FD);

/// Return true iff a structure contains only empty base classes (per \ref
/// isEmptyRecordForLayout) and fields (per \ref isEmptyFieldForLayout).  Note,
/// C++ record fields are considered empty if the type is empty, so this is
/// not the same as \ref isEmptyRecord.
bool isEmptyRecordForLayout(const ASTContext &Ctx, QualType T);

/// The Itanium base layout rule allows virtual bases to overlap other bases,
/// which complicates layout in specific ways.
///
/// Note specifically that the ms_struct attribute doesn't change this.
bool isOverlappingVBaseABI(const ASTContext &Ctx);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_RECORDLAYOUTUTILS_H
