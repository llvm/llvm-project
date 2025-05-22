//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// Defines the interface to generically handle CIR floating-point types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INCLUDE_CLANG_CIR_INTERFACES_CIRFPTYPEINTERFACE_H
#define LLVM_CLANG_INCLUDE_CLANG_CIR_INTERFACES_CIRFPTYPEINTERFACE_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"

/// Include the tablegen'd interface declarations.
#include "clang/CIR/Interfaces/CIRFPTypeInterface.h.inc"

#endif // LLVM_CLANG_INCLUDE_CLANG_CIR_INTERFACES_CIRFPTYPEINTERFACE_H
