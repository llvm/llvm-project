//===-- Utils.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"

fir::TypeInfoOp fir::lookupTypeInfoOp(fir::RecordType recordType,
                                      mlir::ModuleOp module,
                                      const mlir::SymbolTable *symbolTable) {
  // fir.type_info was created with the mangled name of the derived type.
  // It is the same as the name in the related fir.type, except when a pass
  // lowered the fir.type (e.g., when lowering fir.boxproc type if the type has
  // pointer procedure components), in which case suffix may have been added to
  // the fir.type name. Get rid of them when looking up for the fir.type_info.
  llvm::StringRef originalMangledTypeName =
      fir::NameUniquer::dropTypeConversionMarkers(recordType.getName());
  return fir::lookupTypeInfoOp(originalMangledTypeName, module, symbolTable);
}

fir::TypeInfoOp fir::lookupTypeInfoOp(llvm::StringRef name,
                                      mlir::ModuleOp module,
                                      const mlir::SymbolTable *symbolTable) {
  if (symbolTable)
    if (auto typeInfo = symbolTable->lookup<fir::TypeInfoOp>(name))
      return typeInfo;
  return module.lookupSymbol<fir::TypeInfoOp>(name);
}

std::optional<llvm::ArrayRef<int64_t>> fir::getComponentLowerBoundsIfNonDefault(
    fir::RecordType recordType, llvm::StringRef component,
    mlir::ModuleOp module, const mlir::SymbolTable *symbolTable) {
  fir::TypeInfoOp typeInfo =
      fir::lookupTypeInfoOp(recordType, module, symbolTable);
  if (!typeInfo || typeInfo.getComponentInfo().empty())
    return std::nullopt;
  for (auto componentInfo :
       typeInfo.getComponentInfo().getOps<fir::DTComponentOp>())
    if (componentInfo.getName() == component)
      return componentInfo.getLowerBounds();
  return std::nullopt;
}
