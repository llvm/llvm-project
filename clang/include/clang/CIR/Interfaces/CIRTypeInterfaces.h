//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// Defines cir type interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_INTERFACES_CIRTYPEINTERFACES_H
#define CLANG_CIR_INTERFACES_CIRTYPEINTERFACES_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"

/// Include the tablegen'd interface declarations.
#include "clang/CIR/Interfaces/CIRTypeInterfaces.h.inc"

#endif // CLANG_CIR_INTERFACES_CIRTYPEINTERFACES_H
