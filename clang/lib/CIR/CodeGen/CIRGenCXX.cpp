//===----------------------------------------------------------------------===//
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

#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

cir::FuncOp CIRGenModule::codegenCXXStructor(GlobalDecl gd) {
  const CIRGenFunctionInfo &fnInfo =
      getTypes().arrangeCXXStructorDeclaration(gd);
  cir::FuncType funcType = getTypes().getFunctionType(fnInfo);
  cir::FuncOp fn = getAddrOfCXXStructor(gd, &fnInfo, /*FnType=*/nullptr,
                                        /*DontDefer=*/true, ForDefinition);
  assert(!cir::MissingFeatures::opFuncLinkage());
  CIRGenFunction cgf{*this, builder};
  curCGF = &cgf;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    cgf.generateCode(gd, fn, funcType);
  }
  curCGF = nullptr;

  setNonAliasAttributes(gd, fn);
  assert(!cir::MissingFeatures::opFuncAttributesForDefinition());
  return fn;
}
