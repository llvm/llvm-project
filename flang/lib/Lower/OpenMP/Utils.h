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
namespace Fortran {

namespace semantics {
class Symbol;
} // namespace semantics

namespace parser {
struct OmpObject;
struct OmpObjectList;
} // namespace parser

namespace lower {
namespace pft {
struct Evaluation;
}

class AbstractConverter;

namespace omp {

using DeclareTargetCapturePair =
    std::pair<mlir::omp::DeclareTargetCaptureClause, const semantics::Symbol &>;

// A small helper structure for keeping track of a component members MapInfoOp
// and index data when lowering OpenMP map clauses. Keeps track of the
// placement of the component in the derived type hierarchy it rests within,
// alongside the generated mlir::omp::MapInfoOp for the mapped component.
struct OmpMapMemberIndicesData {
  // The indices representing the component members placement in its derived
  // type parents hierarchy.
  llvm::SmallVector<int> memberPlacementIndices;

  // Placement of the member in the member vector.
  mlir::omp::MapInfoOp memberMap;
};

mlir::omp::MapInfoOp
createMapInfoOp(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::Value baseAddr, mlir::Value varPtrPtr, std::string name,
                mlir::ArrayRef<mlir::Value> bounds,
                mlir::ArrayRef<mlir::Value> members,
                mlir::DenseIntElementsAttr membersIndex, uint64_t mapType,
                mlir::omp::VariableCaptureKind mapCaptureType, mlir::Type retTy,
                bool partialMap = false);

void addChildIndexAndMapToParent(
    const omp::Object &object,
    std::map<const semantics::Symbol *,
             llvm::SmallVector<OmpMapMemberIndicesData>> &parentMemberIndices,
    mlir::omp::MapInfoOp &mapOp, semantics::SemanticsContext &semaCtx);

void insertChildMapInfoIntoParent(
    lower::AbstractConverter &converter,
    std::map<const semantics::Symbol *,
             llvm::SmallVector<OmpMapMemberIndicesData>> &parentMemberIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms,
    llvm::SmallVectorImpl<mlir::Type> *mapSymTypes,
    llvm::SmallVectorImpl<mlir::Location> *mapSymLocs);

mlir::Type getLoopVarType(lower::AbstractConverter &converter,
                          std::size_t loopVarTypeSize);

semantics::Symbol *
getIterationVariableSymbol(const lower::pft::Evaluation &eval);

void gatherFuncAndVarSyms(
    const ObjectList &objects, mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCapturePair> &symbolAndClause);

int64_t getCollapseValue(const List<Clause> &clauses);

semantics::Symbol *getOmpObjectSymbol(const parser::OmpObject &ompObject);

void genObjectList(const ObjectList &objects,
                   lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands);

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMPUTILS_H
