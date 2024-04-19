//===-- Lower/OpenMP/Utils.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_OPENMPUTILS_H
#define FORTRAN_LOWER_OPENMPUTILS_H

#include "Clauses.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<bool> treatIndexAsSection;
extern llvm::cl::opt<bool> enableDelayedPrivatization;

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
              const Fortran::semantics::Symbol &>;

// A small helper structure for keeping track of a
// component members MapInfoOp and index data when
// lowering OpenMP map clauses. Keeps track of the
// placement of the component in the derived type
// hierarchy it rests within, alongside the
// generated mlir::omp::MapInfoOp for the mapped
// component.
struct OmpMapMemberIndicesData {
  // The indices representing the component members
  // placement in it's derived type parents hierarchy
  llvm::SmallVector<int> memberPlacementIndices;

  // placement of the member in the member vector.
  mlir::omp::MapInfoOp memberMap;
};

llvm::SmallVector<int>
generateMemberPlacementIndices(const Object &object,
                               Fortran::semantics::SemanticsContext &semaCtx);

mlir::omp::MapInfoOp
createMapInfoOp(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                mlir::ArrayRef<mlir::Value> bounds,
                mlir::ArrayRef<mlir::Value> members,
                mlir::DenseIntElementsAttr membersIndex, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool partialMap = false);

void insertChildMapInfoIntoParent(
    Fortran::lower::AbstractConverter &converter,
    std::map<const Fortran::semantics::Symbol *,
             llvm::SmallVector<OmpMapMemberIndicesData>> &parentMemberIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs,
    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *mapSymbols);

mlir::Type getLoopVarType(Fortran::lower::AbstractConverter &converter,
                          std::size_t loopVarTypeSize);

void gatherFuncAndVarSyms(
    const ObjectList &objects, mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause);

int64_t getCollapseValue(const List<Clause> &clauses);

Fortran::semantics::Symbol *
getOmpObjectSymbol(const Fortran::parser::OmpObject &ompObject);

void genObjectList(const ObjectList &objects,
                   Fortran::lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands);

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMPUTILS_H
