//===----------------------------------------------------------------------===//
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

#include "CIRGenVTables.h"
#include "CIRGenModule.h"
#include "mlir/IR/Types.h"
#include "clang/AST/VTableBuilder.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;
using namespace clang;
using namespace clang::CIRGen;

CIRGenVTables::CIRGenVTables(CIRGenModule &cgm)
    : cgm(cgm), vtContext(cgm.getASTContext().getVTableContext()) {}

mlir::Type CIRGenModule::getVTableComponentType() {
  mlir::Type ptrTy = builder.getUInt8PtrTy();
  assert(!cir::MissingFeatures::vtableRelativeLayout());
  return ptrTy;
}

mlir::Type CIRGenVTables::getVTableComponentType() {
  return cgm.getVTableComponentType();
}

mlir::Type CIRGenVTables::getVTableType(const VTableLayout &layout) {
  SmallVector<mlir::Type, 4> tys;
  auto componentType = getVTableComponentType();
  for (unsigned i = 0, e = layout.getNumVTables(); i != e; ++i)
    tys.push_back(cir::ArrayType::get(componentType, layout.getVTableSize(i)));

  // FIXME(cir): should VTableLayout be encoded like we do for some
  // AST nodes?
  return cgm.getBuilder().getAnonRecordTy(tys, /*incomplete=*/false);
}
