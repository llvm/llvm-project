//===- FIROpenACCSupportAnalysis.cpp - FIR OpenACCSupport Analysis -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIR-specific OpenACCSupport analysis.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Analysis/FIROpenACCSupportAnalysis.h"

#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace fir {
namespace acc {

std::string FIROpenACCSupportAnalysis::getVariableName(Value v) {
  return fir::acc::getVariableName(v, /*preferDemangledName=*/true);
}

std::string FIROpenACCSupportAnalysis::getRecipeName(mlir::acc::RecipeKind kind,
                                                     Type type, Value var) {
  return fir::acc::getRecipeName(kind, type, var);
}

mlir::InFlightDiagnostic
FIROpenACCSupportAnalysis::emitNYI(Location loc, const Twine &message) {
  TODO(loc, message);
  // Should be unreachable, but we return an actual diagnostic
  // to satisfy the interface.
  return mlir::emitError(loc, "not yet implemented: " + message.str());
}

bool FIROpenACCSupportAnalysis::isValidSymbolUse(Operation *user,
                                                 SymbolRefAttr symbol,
                                                 Operation **definingOpPtr) {
  // First check using the default OpenACC utility (recipes, device globals,
  // acc routine, LLVM intrinsics, declare attribute).
  Operation *definingOp = nullptr;
  if (mlir::acc::isValidSymbolUse(user, symbol, &definingOp)) {
    if (definingOpPtr)
      *definingOpPtr = definingOp;
    return true;
  }

  // Default said no; if we have no defining op, nothing more to check.
  if (!definingOp)
    return false;
  if (definingOpPtr)
    *definingOpPtr = definingOp;

  // Functions marked as Fortran runtime are valid (GPU version expected
  // to be offloaded).
  if (definingOp->hasAttr("fir.runtime"))
    return true;

  // Functions with CUF device/global/host_device attribute are valid.
  if (auto cufProcAttr = definingOp->getAttrOfType<cuf::ProcAttributeAttr>(
          cuf::getProcAttrName())) {
    if (cufProcAttr.getValue() != cuf::ProcAttribute::Host)
      return true;
  }

  return false;
}

bool FIROpenACCSupportAnalysis::isValidValueUse(Value v, Region &region) {
  // First check using the base utility.
  if (mlir::acc::isValidValueUse(v, region))
    return true;

  // FIR-specific: fir.logical is a trivial scalar type that can be
  // passed by value.
  if (mlir::isa<fir::LogicalType>(v.getType()))
    return true;

  return false;
}

std::optional<mlir::acc::TypeSizeAndAlignment>
FIROpenACCSupportAnalysis::getTypeSizeAndAlignment(
    Type ty, ModuleOp module, mlir::acc::OpenACCSupport &support) {
  std::optional<DataLayout> dl = mlir::acc::getDataLayout(module);
  if (!dl)
    return std::nullopt;

  if (isa<fir::ReferenceType, fir::HeapType, fir::LLVMPointerType>(ty))
    return mlir::acc::getTypeSizeAndAlignment(
        LLVM::LLVMPointerType::get(ty.getContext()), module, *dl, &support);

  if (!fir::isa_fir_type(ty))
    return mlir::acc::getTypeSizeAndAlignment(ty, module, *dl, &support);

  fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
                                       /*forceUnifiedTBAATree=*/false, *dl);
  fir::KindMapping kindMap = typeConverter.getKindMap();

  if (auto boxTy = dyn_cast<fir::BaseBoxType>(ty))
    return mlir::acc::getTypeSizeAndAlignment(
        typeConverter.convertBoxTypeAsStruct(boxTy), module, *dl, &support);

  auto sizeAndAlignment = fir::getTypeSizeAndAlignment(
      UnknownLoc::get(ty.getContext()), ty, *dl, kindMap);
  if (!sizeAndAlignment)
    return std::nullopt;

  return mlir::acc::TypeSizeAndAlignment{
      llvm::TypeSize::getFixed(sizeAndAlignment->first),
      llvm::TypeSize::getFixed(sizeAndAlignment->second)};
}

} // namespace acc
} // namespace fir
