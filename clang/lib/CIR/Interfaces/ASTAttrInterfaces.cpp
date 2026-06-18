//====- ASTAttrInterfaces.cpp - Interface to AST Attributes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to AST variable declaration attributes.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Mangle.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"

using namespace cir;

/// Include the generated attribute interfaces.
#include "clang/CIR/Interfaces/ASTAttrInterfaces.cpp.inc"
