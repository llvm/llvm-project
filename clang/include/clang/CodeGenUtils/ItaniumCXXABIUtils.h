//===--- ItaniumCXXABIUtils.h - Shared Itanium C++ ABI queries --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the Itanium C++ ABI queries that both classic CodeGen and
// CIR CodeGen need while lowering ABI constructs whose encoding the ABI
// specifies in terms of the AST.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_ITANIUMCXXABIUTILS_H
#define LLVM_CLANG_CODEGENUTILS_ITANIUMCXXABIUTILS_H

#include "clang/AST/ASTContext.h"

namespace clang::CodeGenUtils {

/// Compute the src2dst_offset hint as described in the Itanium C++ ABI [2.9.7].
CharUnits computeOffsetHint(ASTContext &Ctx, const CXXRecordDecl *Src,
                            const CXXRecordDecl *Dst);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_ITANIUMCXXABIUTILS_H
