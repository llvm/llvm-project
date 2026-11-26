//===- FIROpenACCSupportAnalysis.cpp - FIR OpenACCSupport Analysis --------===//
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

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
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

std::string FIROpenACCSupportAnalysis::getVariableName(
    Value v, mlir::acc::VariableNameConfig config) {
  return fir::acc::getVariableName(v, config.preferDemangledName);
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
  return fir::acc::isValidSymbolUse(user, symbol, definingOpPtr);
}

bool FIROpenACCSupportAnalysis::isValidValueUse(
    Value v, Region &region, SymbolTableCollection *symbolTables) {
  return fir::acc::isValidValueUse(v, region, symbolTables);
}

std::optional<mlir::acc::TypeSizeAndAlignment>
FIROpenACCSupportAnalysis::getTypeSizeAndAlignment(
    Type ty, ModuleOp module, mlir::acc::OpenACCSupport &support) {
  std::optional<DataLayout> dl = mlir::acc::getDataLayout(module);
  if (!dl)
    return std::nullopt;

  if (isa<fir::ReferenceType, fir::PointerType, fir::HeapType,
          fir::LLVMPointerType>(ty))
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

bool FIROpenACCSupportAnalysis::isAddressNotCapturedByCalls(Value v) {
  // F2023 15.5.2.5 paragraph 11: a procedure must not use the address of an
  // actual after returning when that actual is a variable that is neither
  // TARGET nor POINTER, is not accessible from an internal procedure, and
  // cannot be accessed by pending asynchronous input/output or by means
  // outside the program.
  auto varIface = v.getDefiningOp<fir::FortranVariableOpInterface>();
  if (!varIface)
    return false;

  fir::AliasAnalysis aliasAnalysis;
  fir::AliasAnalysis::Source source = aliasAnalysis.getSource(
      v, /*getLastInstantiationPoint=*/true, /*collectScopedOrigins=*/false);

  constexpr fir::FortranVariableFlagsEnum externalAccessFlags =
      fir::FortranVariableFlagsEnum::asynchronous |
      fir::FortranVariableFlagsEnum::fortran_volatile;
  auto hasExternalAccess = [&](fir::FortranVariableOpInterface iface) {
    std::optional<fir::FortranVariableFlagsEnum> attrs =
        iface.getFortranAttrs();
    return attrs && bitEnumContainsAny(*attrs, externalAccessFlags);
  };
  if (hasExternalAccess(varIface))
    return false;
  // A designator such as hlfir.designate usually does not carry the
  // attributes of the variable it designates, so also read them from the
  // declaration that the alias analysis reached.
  if (auto declIface = dyn_cast_or_null<fir::FortranVariableOpInterface>(
          source.origin.instantiationPoint))
    if (hasExternalAccess(declIface))
      return false;

  return source.isFortranUserVariable() && !source.isTargetOrPointer() &&
         source.kind != fir::AliasAnalysis::SourceKind::HostAssoc &&
         !source.isCapturedInInternalProcedure;
}

} // namespace acc
} // namespace fir
