//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CODEGEN_CIRGENCALL_H
#define CLANG_LIB_CODEGEN_CIRGENCALL_H

#include "clang/AST/GlobalDecl.h"
#include "llvm/ADT/SmallVector.h"

namespace clang::CIRGen {

/// Type for representing both the decl and type of parameters to a function.
/// The decl must be either a ParmVarDecl or ImplicitParamDecl.
class FunctionArgList : public llvm::SmallVector<const clang::VarDecl *, 16> {};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CODEGEN_CIRGENCALL_H
