//===--- CGCXX.cpp - Emit LLVM Code for declarations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation.
//
//===----------------------------------------------------------------------===//

// We might split this into multiple files if it gets too unwieldy

#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/GlobalDecl.h"

using namespace clang;
using namespace cir;

mlir::cir::FuncOp CIRGenModule::codegenCXXStructor(GlobalDecl GD) {
  const auto &FnInfo = getTypes().arrangeCXXStructorDeclaration(GD);
  auto Fn = getAddrOfCXXStructor(GD, &FnInfo, /*FnType=*/nullptr,
                                 /*DontDefer=*/true, ForDefinition);

  // TODO: setFunctionLinkage
  CIRGenFunction CGF{*this, builder};
  CurCGF = &CGF;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    CGF.generateCode(GD, Fn, FnInfo);
  }
  CurCGF = nullptr;

  // TODO: setNonAliasAttributes
  // TODO: SetLLVMFunctionAttributesForDefinition
  return Fn;
}
