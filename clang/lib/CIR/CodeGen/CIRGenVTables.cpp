//===--- CIRGenVTables.cpp - Emit CIR Code for C++ vtables ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of virtual tables.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "llvm/Support/Format.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cstdio>

using namespace clang;
using namespace cir;

CIRGenVTables::CIRGenVTables(CIRGenModule &CGM)
    : CGM(CGM), VTContext(CGM.getASTContext().getVTableContext()) {}

static bool UseRelativeLayout(const CIRGenModule &CGM) {
  return CGM.getTarget().getCXXABI().isItaniumFamily() &&
         CGM.getItaniumVTableContext().isRelativeLayout();
}

mlir::Type CIRGenModule::getVTableComponentType() {
  mlir::Type ptrTy = builder.getInt8PtrTy();
  if (UseRelativeLayout(*this))
    ptrTy = builder.getInt32PtrTy();
  return ptrTy;
}

mlir::Type CIRGenVTables::getVTableComponentType() {
  return CGM.getVTableComponentType();
}

mlir::Type CIRGenVTables::getVTableType(const VTableLayout &layout) {
  SmallVector<mlir::Type, 4> tys;
  auto ctx = CGM.getBuilder().getContext();
  auto componentType = getVTableComponentType();
  for (unsigned i = 0, e = layout.getNumVTables(); i != e; ++i)
    tys.push_back(
        mlir::cir::ArrayType::get(ctx, componentType, layout.getVTableSize(i)));

  // FIXME(cir): should VTableLayout be encoded like we do for some
  // AST nodes?
  return mlir::cir::StructType::get(ctx, tys, "vtable",
                                    /*body=*/true);
}