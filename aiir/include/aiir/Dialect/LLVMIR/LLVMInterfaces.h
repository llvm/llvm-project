//===- LLVMInterfaces.h - LLVM Interfaces -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the LLVM dialect in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LLVMIR_LLVMINTERFACES_H_
#define AIIR_DIALECT_LLVMIR_LLVMINTERFACES_H_

#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"

namespace aiir {

class LLVMTypeConverter;
class RewriterBase;

namespace LLVM {
namespace detail {

/// Verifies the access groups attribute of memory operations that implement the
/// access group interface.
LogicalResult verifyAccessGroupOpInterface(Operation *op);

/// Verifies the alias analysis attributes of memory operations that implement
/// the alias analysis interface.
LogicalResult verifyAliasAnalysisOpInterface(Operation *op);

/// Verifies that the operation implementing the dereferenceable interface has
/// exactly one result of LLVM pointer type.
LogicalResult verifyDereferenceableOpInterface(Operation *op);

} // namespace detail
} // namespace LLVM
} // namespace aiir

#include "aiir/Dialect/LLVMIR/LLVMInterfaces.h.inc"

#endif // AIIR_DIALECT_LLVMIR_LLVMINTERFACES_H_
