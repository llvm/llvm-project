//===-- CIRGenOpenCLRuntime.cpp - Interface to OpenCL Runtimes ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for OpenCL CIR generation. Concrete
// subclasses of this implement code generation for specific OpenCL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CIRGenOpenCLRuntime.h"
#include "CIRGenFunction.h"

#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

using namespace clang;
using namespace cir;

CIRGenOpenCLRuntime::~CIRGenOpenCLRuntime() {}

void CIRGenOpenCLRuntime::buildWorkGroupLocalVarDecl(CIRGenFunction &CGF,
                                                     const VarDecl &D) {
  return CGF.buildStaticVarDecl(D,
                                mlir::cir::GlobalLinkageKind::InternalLinkage);
}
