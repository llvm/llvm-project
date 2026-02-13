//===- ASTAttrInterfaces.h - CIR AST Interfaces -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_INTERFACES_ASTATTRINTERFACES_H
#define CLANG_CIR_INTERFACES_ASTATTRINTERFACES_H

#include "mlir/IR/Attributes.h"

#include "clang/AST/Attr.h"
#include "clang/AST/DeclTemplate.h"

/// Include the generated interface declarations.
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h.inc"

#endif // CLANG_CIR_INTERFACES_ASTATTRINTERFACES_H
