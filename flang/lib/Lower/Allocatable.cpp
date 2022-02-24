//===-- Allocatable.cpp -- Allocatable statements lowering ----------------===//
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

#include "flang/Lower/Allocatable.h"
#include "flang/Evaluate/tools.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "llvm/Support/CommandLine.h"

/// By default fir memory operation fir::AllocMemOp/fir::FreeMemOp are used.
/// This switch allow forcing the use of runtime and descriptors for everything.
/// This is mainly intended as a debug switch.
static llvm::cl::opt<bool> useAllocateRuntime(
    "use-alloc-runtime",
    llvm::cl::desc("Lower allocations to fortran runtime calls"),
    llvm::cl::init(false));
/// Switch to force lowering of allocatable and pointers to descriptors in all
/// cases for debug purposes.
static llvm::cl::opt<bool> useDescForMutableBox(
    "use-desc-for-alloc",
    llvm::cl::desc("Always use descriptors for POINTER and ALLOCATABLE"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// MutableBoxValue creation implementation
//===----------------------------------------------------------------------===//

/// Is this symbol a pointer to a pointer array that does not have the
/// CONTIGUOUS attribute ?
static inline bool
isNonContiguousArrayPointer(const Fortran::semantics::Symbol &sym) {
  return Fortran::semantics::IsPointer(sym) && sym.Rank() != 0 &&
         !sym.attrs().test(Fortran::semantics::Attr::CONTIGUOUS);
}

/// Is this a local procedure symbol in a procedure that contains internal
/// procedures ?
static bool mayBeCapturedInInternalProc(const Fortran::semantics::Symbol &sym) {
  const Fortran::semantics::Scope &owner = sym.owner();
  Fortran::semantics::Scope::Kind kind = owner.kind();
  // Test if this is a procedure scope that contains a subprogram scope that is
  // not an interface.
  if (kind == Fortran::semantics::Scope::Kind::Subprogram ||
      kind == Fortran::semantics::Scope::Kind::MainProgram)
    for (const Fortran::semantics::Scope &childScope : owner.children())
      if (childScope.kind() == Fortran::semantics::Scope::Kind::Subprogram)
        if (const Fortran::semantics::Symbol *childSym = childScope.symbol())
          if (const auto *details =
                  childSym->detailsIf<Fortran::semantics::SubprogramDetails>())
            if (!details->isInterface())
              return true;
  return false;
}

/// In case it is safe to track the properties in variables outside a
/// descriptor, create the variables to hold the mutable properties of the
/// entity var. The variables are not initialized here.
static fir::MutableProperties
createMutableProperties(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc,
                        const Fortran::lower::pft::Variable &var,
                        mlir::ValueRange nonDeferredParams) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  const Fortran::semantics::Symbol &sym = var.getSymbol();
  // Globals and dummies may be associated, creating local variables would
  // require keeping the values and descriptor before and after every single
  // impure calls in the current scope (not only the ones taking the variable as
  // arguments. All.) Volatile means the variable may change in ways not defined
  // per Fortran, so lowering can most likely not keep the descriptor and values
  // in sync as needed.
  // Pointers to non contiguous arrays need to be represented with a fir.box to
  // account for the discontiguity.
  // Pointer/Allocatable in internal procedure are descriptors in the host link,
  // and it would increase complexity to sync this descriptor with the local
  // values every time the host link is escaping.
  if (var.isGlobal() || Fortran::semantics::IsDummy(sym) ||
      Fortran::semantics::IsFunctionResult(sym) ||
      sym.attrs().test(Fortran::semantics::Attr::VOLATILE) ||
      isNonContiguousArrayPointer(sym) || useAllocateRuntime ||
      useDescForMutableBox || mayBeCapturedInInternalProc(sym))
    return {};
  fir::MutableProperties mutableProperties;
  std::string name = converter.mangleName(sym);
  mlir::Type baseAddrTy = converter.genType(sym);
  if (auto boxType = baseAddrTy.dyn_cast<fir::BoxType>())
    baseAddrTy = boxType.getEleTy();
  // Allocate and set a variable to hold the address.
  // It will be set to null in setUnallocatedStatus.
  mutableProperties.addr =
      builder.allocateLocal(loc, baseAddrTy, name + ".addr", "",
                            /*shape=*/llvm::None, /*typeparams=*/llvm::None);
  // Allocate variables to hold lower bounds and extents.
  int rank = sym.Rank();
  mlir::Type idxTy = builder.getIndexType();
  for (decltype(rank) i = 0; i < rank; ++i) {
    mlir::Value lboundVar =
        builder.allocateLocal(loc, idxTy, name + ".lb" + std::to_string(i), "",
                              /*shape=*/llvm::None, /*typeparams=*/llvm::None);
    mlir::Value extentVar =
        builder.allocateLocal(loc, idxTy, name + ".ext" + std::to_string(i), "",
                              /*shape=*/llvm::None, /*typeparams=*/llvm::None);
    mutableProperties.lbounds.emplace_back(lboundVar);
    mutableProperties.extents.emplace_back(extentVar);
  }

  // Allocate variable to hold deferred length parameters.
  mlir::Type eleTy = baseAddrTy;
  if (auto newTy = fir::dyn_cast_ptrEleTy(eleTy))
    eleTy = newTy;
  if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>())
    eleTy = seqTy.getEleTy();
  if (auto record = eleTy.dyn_cast<fir::RecordType>())
    if (record.getNumLenParams() != 0)
      TODO(loc, "deferred length type parameters.");
  if (fir::isa_char(eleTy) && nonDeferredParams.empty()) {
    mlir::Value lenVar =
        builder.allocateLocal(loc, builder.getCharacterLengthType(),
                              name + ".len", "", /*shape=*/llvm::None,
                              /*typeparams=*/llvm::None);
    mutableProperties.deferredParams.emplace_back(lenVar);
  }
  return mutableProperties;
}

fir::MutableBoxValue Fortran::lower::createMutableBox(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    const Fortran::lower::pft::Variable &var, mlir::Value boxAddr,
    mlir::ValueRange nonDeferredParams) {

  fir::MutableProperties mutableProperties =
      createMutableProperties(converter, loc, var, nonDeferredParams);
  fir::MutableBoxValue box(boxAddr, nonDeferredParams, mutableProperties);
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  if (!var.isGlobal() && !Fortran::semantics::IsDummy(var.getSymbol()))
    fir::factory::disassociateMutableBox(builder, loc, box);
  return box;
}
