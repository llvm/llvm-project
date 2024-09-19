//===--- CIRGenTBAA.h - TBAA information for LLVM CIRGen --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that manages TBAA information and defines the TBAA policy
// for the optimizer to use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENTBAA_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENTBAA_H

namespace cir {

// TBAAAccessInfo - Describes a memory access in terms of TBAA.
struct TBAAAccessInfo {};

/// CIRGenTBAA - This class organizes the cross-module state that is used while
/// lowering AST types to LLVM types.
class CIRGenTBAA {};

} // namespace cir

#endif
