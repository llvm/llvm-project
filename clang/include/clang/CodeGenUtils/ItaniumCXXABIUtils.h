//===--- ItaniumCXXABIUtils.h - Shared Itanium C++ ABI queries --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the Itanium C++ ABI type_info flag values, together with
// the queries that both classic CodeGen and CIR CodeGen need while lowering
// ABI constructs whose encoding the ABI specifies in terms of the AST.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_ITANIUMCXXABIUTILS_H
#define LLVM_CLANG_CODEGENUTILS_ITANIUMCXXABIUTILS_H

#include "clang/AST/ASTContext.h"

namespace clang::CodeGenUtils {

/// Flags for abi::__pbase_type_info.
enum PointerTypeInfoFlags {
  /// PTI_Const - Type has const qualifier.
  PTI_Const = 0x1,

  /// PTI_Volatile - Type has volatile qualifier.
  PTI_Volatile = 0x2,

  /// PTI_Restrict - Type has restrict qualifier.
  PTI_Restrict = 0x4,

  /// PTI_Incomplete - Type is incomplete.
  PTI_Incomplete = 0x8,

  /// PTI_ContainingClassIncomplete - Containing class is incomplete.
  /// (in pointer to member).
  PTI_ContainingClassIncomplete = 0x10,

  /// PTI_TransactionSafe - Pointee is transaction_safe function (C++ TM TS).
  // PTI_TransactionSafe = 0x20,

  /// PTI_Noexcept - Pointee is noexcept function (C++1z).
  PTI_Noexcept = 0x40,
};

/// Compute the src2dst_offset hint as described in the Itanium C++ ABI [2.9.7].
CharUnits computeOffsetHint(ASTContext &Ctx, const CXXRecordDecl *Src,
                            const CXXRecordDecl *Dst);

/// Returns whether the given type contains an incomplete class type.  This is
/// true if
///
///   * The given type is an incomplete class type.
///   * The given type is a pointer type whose pointee type contains an
///     incomplete class type.
///   * The given type is a member pointer type whose class is an incomplete
///     class type.
///   * The given type is a member pointer type whose pointee type contains an
///     incomplete class type.
bool containsIncompleteClassType(QualType Ty);

/// Compute the flags for a __pbase_type_info, and remove the corresponding
/// pieces from \p Type.
unsigned extractPBaseFlags(const ASTContext &Ctx, QualType &Type);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_ITANIUMCXXABIUTILS_H
