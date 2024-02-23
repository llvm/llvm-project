//===-- Lower/OpenMP/Utils.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_OPENMPUTILS_H
#define FORTRAN_LOWER_OPENMPUTILS_H

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<bool> treatIndexAsSection;

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace llvm::omp {
enum class OpenMPOffloadMappingFlags : uint64_t;
} // namespace llvm::omp

namespace Fortran {

namespace semantics {
class Symbol;
} // namespace semantics

namespace parser {
struct OmpObject;
struct OmpObjectList;
} // namespace parser

namespace lower {

class AbstractConverter;

namespace omp {

using DeclareTargetCapturePair =
    std::pair<mlir::omp::DeclareTargetCaptureClause,
              Fortran::semantics::Symbol>;

mlir::omp::MapInfoOp
createMapInfoOp(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                mlir::SmallVector<mlir::Value> bounds,
                mlir::SmallVector<mlir::Value> members,
                mlir::ArrayAttr membersIndex, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool partialMap = false);

void checkAndApplyDeclTargetMapFlags(
    Fortran::lower::AbstractConverter &converter,
    llvm::omp::OpenMPOffloadMappingFlags &mapFlags,
    const Fortran::semantics::Symbol &symbol);

int findComponentMemberPlacement(
    const Fortran::semantics::Symbol *dTypeSym,
    const Fortran::semantics::Symbol *componentSym);

void insertChildMapInfoIntoParent(
    Fortran::lower::AbstractConverter &converter,
    llvm::SmallVector<const Fortran::semantics::Symbol *> &memberParentSyms,
    llvm::SmallVector<mlir::omp::MapInfoOp> &memberMaps,
    llvm::SmallVector<mlir::Attribute> &memberPlacementIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *mapSymbols);
    
void gatherFuncAndVarSyms(
    const Fortran::parser::OmpObjectList &objList,
    mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause);

Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject);

void genObjectList(const Fortran::parser::OmpObjectList &objectList,
                   Fortran::lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands);

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMPUTILS_H
