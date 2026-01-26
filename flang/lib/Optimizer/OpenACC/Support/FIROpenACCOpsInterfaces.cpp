//===-- FIROpenACCOpsInterfaces.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of external operation interfaces for FIR.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Support/FIROpenACCOpsInterfaces.h"

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallSet.h"

namespace fir::acc {

template <>
mlir::Value PartialEntityAccessModel<fir::ArrayCoorOp>::getBaseEntity(
    mlir::Operation *op) const {
  return mlir::cast<fir::ArrayCoorOp>(op).getMemref();
}

template <>
mlir::Value PartialEntityAccessModel<fir::CoordinateOp>::getBaseEntity(
    mlir::Operation *op) const {
  return mlir::cast<fir::CoordinateOp>(op).getRef();
}

template <>
mlir::Value PartialEntityAccessModel<hlfir::DesignateOp>::getBaseEntity(
    mlir::Operation *op) const {
  return mlir::cast<hlfir::DesignateOp>(op).getMemref();
}

mlir::Value PartialEntityAccessModel<fir::DeclareOp>::getBaseEntity(
    mlir::Operation *op) const {
  auto declareOp = mlir::cast<fir::DeclareOp>(op);
  // If storage is present, return it (partial view case)
  if (mlir::Value storage = declareOp.getStorage())
    return storage;
  // Otherwise return the memref (complete view case)
  return declareOp.getMemref();
}

bool PartialEntityAccessModel<fir::DeclareOp>::isCompleteView(
    mlir::Operation *op) const {
  // Complete view if storage is absent
  return !mlir::cast<fir::DeclareOp>(op).getStorage();
}

mlir::Value PartialEntityAccessModel<hlfir::DeclareOp>::getBaseEntity(
    mlir::Operation *op) const {
  auto declareOp = mlir::cast<hlfir::DeclareOp>(op);
  // If storage is present, return it (partial view case)
  if (mlir::Value storage = declareOp.getStorage())
    return storage;
  // Otherwise return the memref (complete view case)
  return declareOp.getMemref();
}

bool PartialEntityAccessModel<hlfir::DeclareOp>::isCompleteView(
    mlir::Operation *op) const {
  // Complete view if storage is absent
  return !mlir::cast<hlfir::DeclareOp>(op).getStorage();
}

mlir::SymbolRefAttr AddressOfGlobalModel::getSymbol(mlir::Operation *op) const {
  return mlir::cast<fir::AddrOfOp>(op).getSymbolAttr();
}

bool GlobalVariableModel::isConstant(mlir::Operation *op) const {
  auto globalOp = mlir::cast<fir::GlobalOp>(op);
  return globalOp.getConstant().has_value();
}

mlir::Region *GlobalVariableModel::getInitRegion(mlir::Operation *op) const {
  auto globalOp = mlir::cast<fir::GlobalOp>(op);
  return globalOp.hasInitializationBody() ? &globalOp.getRegion() : nullptr;
}

bool GlobalVariableModel::isDeviceData(mlir::Operation *op) const {
  if (auto dataAttr = cuf::getDataAttr(op))
    return cuf::isDeviceDataAttribute(dataAttr.getValue());
  return false;
}

// Helper to recursively process address-of operations in derived type
// descriptors and collect all needed fir.globals.
static void processAddrOfOpInDerivedTypeDescriptor(
    fir::AddrOfOp addrOfOp, mlir::SymbolTable &symTab,
    llvm::SmallSet<mlir::Operation *, 16> &globalsSet,
    llvm::SmallVectorImpl<mlir::SymbolRefAttr> &symbols) {
  if (auto globalOp = symTab.lookup<fir::GlobalOp>(
          addrOfOp.getSymbol().getLeafReference().getValue())) {
    if (globalsSet.contains(globalOp))
      return;
    globalsSet.insert(globalOp);
    symbols.push_back(addrOfOp.getSymbolAttr());
    globalOp.walk([&](fir::AddrOfOp op) {
      processAddrOfOpInDerivedTypeDescriptor(op, symTab, globalsSet, symbols);
    });
  }
}

// Utility to collect referenced symbols for type descriptors of derived types.
// This is the common logic for operations that may require type descriptor
// globals.
static void collectReferencedSymbolsForType(
    mlir::Type ty, mlir::Operation *op,
    llvm::SmallVectorImpl<mlir::SymbolRefAttr> &symbols,
    mlir::SymbolTable *symbolTable) {
  ty = fir::getDerivedType(fir::unwrapRefType(ty));

  // Look for type descriptor globals only if it's a derived (record) type
  if (auto recTy = mlir::dyn_cast_if_present<fir::RecordType>(ty)) {
    // If no symbol table provided, simply add the type descriptor name
    if (!symbolTable) {
      symbols.push_back(mlir::SymbolRefAttr::get(
          op->getContext(),
          fir::NameUniquer::getTypeDescriptorName(recTy.getName())));
      return;
    }

    // Otherwise, do full lookup and recursive processing
    llvm::SmallSet<mlir::Operation *, 16> globalsSet;

    fir::GlobalOp globalOp = symbolTable->lookup<fir::GlobalOp>(
        fir::NameUniquer::getTypeDescriptorName(recTy.getName()));
    if (!globalOp)
      globalOp = symbolTable->lookup<fir::GlobalOp>(
          fir::NameUniquer::getTypeDescriptorAssemblyName(recTy.getName()));

    if (globalOp) {
      globalsSet.insert(globalOp);
      symbols.push_back(
          mlir::SymbolRefAttr::get(op->getContext(), globalOp.getSymName()));
      globalOp.walk([&](fir::AddrOfOp addrOp) {
        processAddrOfOpInDerivedTypeDescriptor(addrOp, *symbolTable, globalsSet,
                                               symbols);
      });
    }
  }
}

template <>
void IndirectGlobalAccessModel<fir::AllocaOp>::getReferencedSymbols(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::SymbolRefAttr> &symbols,
    mlir::SymbolTable *symbolTable) const {
  auto allocaOp = mlir::cast<fir::AllocaOp>(op);
  collectReferencedSymbolsForType(allocaOp.getType(), op, symbols, symbolTable);
}

template <>
void IndirectGlobalAccessModel<fir::EmboxOp>::getReferencedSymbols(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::SymbolRefAttr> &symbols,
    mlir::SymbolTable *symbolTable) const {
  auto emboxOp = mlir::cast<fir::EmboxOp>(op);
  collectReferencedSymbolsForType(emboxOp.getMemref().getType(), op, symbols,
                                  symbolTable);
}

template <>
void IndirectGlobalAccessModel<fir::ReboxOp>::getReferencedSymbols(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::SymbolRefAttr> &symbols,
    mlir::SymbolTable *symbolTable) const {
  auto reboxOp = mlir::cast<fir::ReboxOp>(op);
  collectReferencedSymbolsForType(reboxOp.getBox().getType(), op, symbols,
                                  symbolTable);
}

template <>
void IndirectGlobalAccessModel<fir::TypeDescOp>::getReferencedSymbols(
    mlir::Operation *op, llvm::SmallVectorImpl<mlir::SymbolRefAttr> &symbols,
    mlir::SymbolTable *symbolTable) const {
  auto typeDescOp = mlir::cast<fir::TypeDescOp>(op);
  collectReferencedSymbolsForType(typeDescOp.getInType(), op, symbols,
                                  symbolTable);
}

template <>
bool OperationMoveModel<mlir::acc::LoopOp>::canMoveFromDescendant(
    mlir::Operation *op, mlir::Operation *descendant,
    mlir::Operation *candidate) const {
  // It should be always allowed to move operations from descendants
  // of acc.loop into the acc.loop.
  return true;
}

template <>
bool OperationMoveModel<mlir::acc::LoopOp>::canMoveOutOf(
    mlir::Operation *op, mlir::Operation *candidate) const {
  // Disallow moving operations, which have operands that are referenced
  // in the data operands (e.g. in [first]private() etc.) of the acc.loop.
  // For example:
  //   %17 = acc.private var(%16 : !fir.box<!fir.array<?xf32>>)
  //   acc.loop private(%17 : !fir.box<!fir.array<?xf32>>) ... {
  //     %19 = fir.box_addr %17
  //   }
  // We cannot hoist %19 without violating assumptions that OpenACC
  // transformations rely on.

  // In general, some movement out of acc.loop is allowed,
  // so return true if candidate is nullptr.
  if (!candidate)
    return true;

  auto loopOp = mlir::cast<mlir::acc::LoopOp>(op);
  unsigned numDataOperands = loopOp.getNumDataOperands();
  for (unsigned i = 0; i < numDataOperands; ++i) {
    mlir::Value dataOperand = loopOp.getDataOperand(i);
    return !llvm::any_of(candidate->getOperands(),
                         [&](mlir::Value candidateOperand) {
                           return dataOperand == candidateOperand;
                         });
  }
  return true;
}

} // namespace fir::acc
