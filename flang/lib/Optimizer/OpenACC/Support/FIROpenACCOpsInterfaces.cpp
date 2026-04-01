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
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallSet.h"

namespace fir::acc {

aiir::Value ReductionInitOpFortranObjectViewModel::getViewSource(
    aiir::Operation *op, aiir::OpResult resultView) const {
  assert(resultView.getOwner() == op && "result value must be the op's result");
  assert(op->getNumResults() == 1 &&
         "definition of acc.reduction_init changed");
  auto iface = aiir::cast<aiir::RegionBranchOpInterface>(op);
  llvm::SmallVector<aiir::Value, 1> resultValues;
  iface.getPredecessorValues(aiir::RegionSuccessor::parent(), /*index=*/0,
                             resultValues);
  assert(!resultValues.empty() &&
         "acc.reduction_init's result must have at least one possible value");
  aiir::Value passThroughValue;
  for (aiir::Value v : resultValues) {
    if (!passThroughValue) {
      passThroughValue = v;
      continue;
    }
    assert(passThroughValue == v &&
           "acc.reduction_init must return the same allocation");
  }
  return passThroughValue;
}

std::optional<std::int64_t>
ReductionInitOpFortranObjectViewModel::getViewOffset(
    aiir::Operation *op, aiir::OpResult resultView) const {
  assert(resultView.getOwner() == op && "result value must be the op's result");
  return 0;
}

template <>
aiir::Value PartialEntityAccessModel<fir::ArrayCoorOp>::getBaseEntity(
    aiir::Operation *op) const {
  return aiir::cast<fir::ArrayCoorOp>(op).getMemref();
}

template <>
aiir::Value PartialEntityAccessModel<fir::CoordinateOp>::getBaseEntity(
    aiir::Operation *op) const {
  return aiir::cast<fir::CoordinateOp>(op).getRef();
}

template <>
aiir::Value PartialEntityAccessModel<hlfir::DesignateOp>::getBaseEntity(
    aiir::Operation *op) const {
  return aiir::cast<hlfir::DesignateOp>(op).getMemref();
}

aiir::Value PartialEntityAccessModel<fir::DeclareOp>::getBaseEntity(
    aiir::Operation *op) const {
  auto declareOp = aiir::cast<fir::DeclareOp>(op);
  // If storage is present, return it (partial view case)
  if (aiir::Value storage = declareOp.getStorage())
    return storage;
  // Otherwise return the memref (complete view case)
  return declareOp.getMemref();
}

bool PartialEntityAccessModel<fir::DeclareOp>::isCompleteView(
    aiir::Operation *op) const {
  // Complete view if storage is absent
  return !aiir::cast<fir::DeclareOp>(op).getStorage();
}

aiir::Value PartialEntityAccessModel<hlfir::DeclareOp>::getBaseEntity(
    aiir::Operation *op) const {
  auto declareOp = aiir::cast<hlfir::DeclareOp>(op);
  // If storage is present, return it (partial view case)
  if (aiir::Value storage = declareOp.getStorage())
    return storage;
  // Otherwise return the memref (complete view case)
  return declareOp.getMemref();
}

bool PartialEntityAccessModel<hlfir::DeclareOp>::isCompleteView(
    aiir::Operation *op) const {
  // Complete view if storage is absent
  return !aiir::cast<hlfir::DeclareOp>(op).getStorage();
}

aiir::SymbolRefAttr AddressOfGlobalModel::getSymbol(aiir::Operation *op) const {
  return aiir::cast<fir::AddrOfOp>(op).getSymbolAttr();
}

bool GlobalVariableModel::isConstant(aiir::Operation *op) const {
  auto globalOp = aiir::cast<fir::GlobalOp>(op);
  return globalOp.getConstant().has_value();
}

aiir::Region *GlobalVariableModel::getInitRegion(aiir::Operation *op) const {
  auto globalOp = aiir::cast<fir::GlobalOp>(op);
  return globalOp.hasInitializationBody() ? &globalOp.getRegion() : nullptr;
}

bool GlobalVariableModel::isDeviceData(aiir::Operation *op) const {
  if (auto dataAttr = cuf::getDataAttr(op))
    return cuf::isDeviceDataAttribute(dataAttr.getValue());
  return false;
}

// Helper to recursively process address-of operations in derived type
// descriptors and collect all needed fir.globals.
static void processAddrOfOpInDerivedTypeDescriptor(
    fir::AddrOfOp addrOfOp, aiir::SymbolTable &symTab,
    llvm::SmallSet<aiir::Operation *, 16> &globalsSet,
    llvm::SmallVectorImpl<aiir::SymbolRefAttr> &symbols) {
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
    aiir::Type ty, aiir::Operation *op,
    llvm::SmallVectorImpl<aiir::SymbolRefAttr> &symbols,
    aiir::SymbolTable *symbolTable) {
  ty = fir::getDerivedType(fir::unwrapRefType(ty));

  // Look for type descriptor globals only if it's a derived (record) type
  if (auto recTy = aiir::dyn_cast_if_present<fir::RecordType>(ty)) {
    // If no symbol table provided, simply add the type descriptor name
    if (!symbolTable) {
      symbols.push_back(aiir::SymbolRefAttr::get(
          op->getContext(),
          fir::NameUniquer::getTypeDescriptorName(recTy.getName())));
      return;
    }

    // Otherwise, do full lookup and recursive processing
    llvm::SmallSet<aiir::Operation *, 16> globalsSet;

    fir::GlobalOp globalOp = symbolTable->lookup<fir::GlobalOp>(
        fir::NameUniquer::getTypeDescriptorName(recTy.getName()));
    if (!globalOp)
      globalOp = symbolTable->lookup<fir::GlobalOp>(
          fir::NameUniquer::getTypeDescriptorAssemblyName(recTy.getName()));

    if (globalOp) {
      globalsSet.insert(globalOp);
      symbols.push_back(
          aiir::SymbolRefAttr::get(op->getContext(), globalOp.getSymName()));
      globalOp.walk([&](fir::AddrOfOp addrOp) {
        processAddrOfOpInDerivedTypeDescriptor(addrOp, *symbolTable, globalsSet,
                                               symbols);
      });
    }
  }
}

template <>
void IndirectGlobalAccessModel<fir::AllocaOp>::getReferencedSymbols(
    aiir::Operation *op, llvm::SmallVectorImpl<aiir::SymbolRefAttr> &symbols,
    aiir::SymbolTable *symbolTable) const {
  auto allocaOp = aiir::cast<fir::AllocaOp>(op);
  collectReferencedSymbolsForType(allocaOp.getType(), op, symbols, symbolTable);
}

template <>
void IndirectGlobalAccessModel<fir::EmboxOp>::getReferencedSymbols(
    aiir::Operation *op, llvm::SmallVectorImpl<aiir::SymbolRefAttr> &symbols,
    aiir::SymbolTable *symbolTable) const {
  auto emboxOp = aiir::cast<fir::EmboxOp>(op);
  collectReferencedSymbolsForType(emboxOp.getMemref().getType(), op, symbols,
                                  symbolTable);
}

template <>
void IndirectGlobalAccessModel<fir::ReboxOp>::getReferencedSymbols(
    aiir::Operation *op, llvm::SmallVectorImpl<aiir::SymbolRefAttr> &symbols,
    aiir::SymbolTable *symbolTable) const {
  auto reboxOp = aiir::cast<fir::ReboxOp>(op);
  collectReferencedSymbolsForType(reboxOp.getBox().getType(), op, symbols,
                                  symbolTable);
}

template <>
void IndirectGlobalAccessModel<fir::TypeDescOp>::getReferencedSymbols(
    aiir::Operation *op, llvm::SmallVectorImpl<aiir::SymbolRefAttr> &symbols,
    aiir::SymbolTable *symbolTable) const {
  auto typeDescOp = aiir::cast<fir::TypeDescOp>(op);
  collectReferencedSymbolsForType(typeDescOp.getInType(), op, symbols,
                                  symbolTable);
}

template <>
void IndirectGlobalAccessModel<fir::UseStmtOp>::getReferencedSymbols(
    aiir::Operation *op, llvm::SmallVectorImpl<aiir::SymbolRefAttr> &symbols,
    aiir::SymbolTable *symbolTable) const {
  auto useStmtOp = aiir::cast<fir::UseStmtOp>(op);
  if (auto onlySymbols = useStmtOp.getOnlySymbols()) {
    for (auto attr : *onlySymbols)
      if (auto symRef = aiir::dyn_cast<aiir::SymbolRefAttr>(attr))
        symbols.push_back(symRef);
  }
  if (auto renames = useStmtOp.getRenames()) {
    for (auto attr : *renames)
      if (auto renameAttr = aiir::dyn_cast<fir::UseRenameAttr>(attr))
        symbols.push_back(renameAttr.getSymbol());
  }
}

template <>
bool OperationMoveModel<aiir::acc::LoopOp>::canMoveFromDescendant(
    aiir::Operation *op, aiir::Operation *descendant,
    aiir::Operation *candidate) const {
  // It should be always allowed to move operations from descendants
  // of acc.loop into the acc.loop.
  return true;
}

template <>
bool OperationMoveModel<aiir::acc::LoopOp>::canMoveOutOf(
    aiir::Operation *op, aiir::Operation *candidate) const {
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

  auto loopOp = aiir::cast<aiir::acc::LoopOp>(op);
  unsigned numDataOperands = loopOp.getNumDataOperands();
  for (unsigned i = 0; i < numDataOperands; ++i) {
    aiir::Value dataOperand = loopOp.getDataOperand(i);
    if (llvm::any_of(candidate->getOperands(),
                     [&](aiir::Value candidateOperand) {
                       return dataOperand == candidateOperand;
                     }))
      return false;
  }
  return true;
}

} // namespace fir::acc
