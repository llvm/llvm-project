//===-- Lower/OpenMP/Utils.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_OPENMPUTILS_H
#define FORTRAN_LOWER_OPENMPUTILS_H

#include "flang/Lower/OpenMP/Clauses.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include <cstdint>

extern llvm::cl::opt<bool> treatIndexAsSection;

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
class StatementContext;
namespace pft {
struct Evaluation;
}

class AbstractConverter;

namespace omp {

struct DeclareTargetCaptureInfo {
  mlir::omp::DeclareTargetCaptureClause clause;
  bool automap = false;
  const semantics::Symbol &symbol;

  DeclareTargetCaptureInfo(mlir::omp::DeclareTargetCaptureClause c,
                           const semantics::Symbol &s, bool a = false)
      : clause(c), automap(a), symbol(s) {}
};

// A small helper structure for keeping track of a component members MapInfoOp
// and index data when lowering OpenMP map clauses. Keeps track of the
// placement of the component in the derived type hierarchy it rests within,
// alongside the generated mlir::omp::MapInfoOp for the mapped component.
//
// As an example of what the contents of this data structure may be like,
// when provided the following derived type and map of that type:
//
// type :: bottom_layer
//   real(8) :: i2
//   real(4) :: array_i2(10)
//   real(4) :: array_j2(10)
// end type bottom_layer
//
// type :: top_layer
//   real(4) :: i
//   integer(4) :: array_i(10)
//   real(4) :: j
//   type(bottom_layer) :: nested
//   integer, allocatable :: array_j(:)
//   integer(4) :: k
// end type top_layer
//
// type(top_layer) :: top_dtype
//
// map(tofrom: top_dtype%nested%i2, top_dtype%k, top_dtype%nested%array_i2)
//
// We would end up with an OmpMapParentAndMemberData populated like below:
//
// memberPlacementIndices:
//  Vector 1: 3, 0
//  Vector 2: 5
//  Vector 3: 3, 1
//
// memberMap:
// Entry 1: omp.map.info for "top_dtype%nested%i2"
// Entry 2: omp.map.info for "top_dtype%k"
// Entry 3: omp.map.info for "top_dtype%nested%array_i2"
//
// And this OmpMapParentAndMemberData would be accessed via the parent
// symbol for top_dtype. Other parent derived type instances that have
// members mapped would have there own OmpMapParentAndMemberData entry
// accessed via their own symbol.
struct OmpMapParentAndMemberData {
  // The indices representing the component members placement in its derived
  // type parents hierarchy.
  llvm::SmallVector<llvm::SmallVector<int64_t>> memberPlacementIndices;

  // Placement of the member in the member vector.
  llvm::SmallVector<mlir::omp::MapInfoOp> memberMap;

  bool isDuplicateMemberMapInfo(llvm::SmallVectorImpl<int64_t> &memberIndices) {
    return llvm::find_if(memberPlacementIndices, [&](auto &memberData) {
             return llvm::equal(memberIndices, memberData);
           }) != memberPlacementIndices.end();
  }

  void addChildIndexAndMapToParent(const omp::Object &object,
                                   mlir::omp::MapInfoOp &mapOp,
                                   semantics::SemanticsContext &semaCtx);
};

void insertChildMapInfoIntoParent(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::StatementContext &stmtCtx,
    std::map<Object, OmpMapParentAndMemberData> &parentMemberIndices,
    llvm::SmallVectorImpl<mlir::Value> &mapOperands,
    llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms);

void generateMemberPlacementIndices(
    const Object &object, llvm::SmallVectorImpl<int64_t> &indices,
    Fortran::semantics::SemanticsContext &semaCtx);

bool isMemberOrParentAllocatableOrPointer(
    const Object &object, Fortran::semantics::SemanticsContext &semaCtx);

mlir::Value createParentSymAndGenIntermediateMaps(
    mlir::Location clauseLocation, Fortran::lower::AbstractConverter &converter,
    semantics::SemanticsContext &semaCtx, lower::StatementContext &stmtCtx,
    omp::ObjectList &objectList, llvm::SmallVectorImpl<int64_t> &indices,
    OmpMapParentAndMemberData &parentMemberIndices, llvm::StringRef asFortran,
    llvm::omp::OpenMPOffloadMappingFlags mapTypeBits);

omp::ObjectList gatherObjectsOf(omp::Object derivedTypeMember,
                                semantics::SemanticsContext &semaCtx);

mlir::Type getLoopVarType(lower::AbstractConverter &converter,
                          std::size_t loopVarTypeSize);

semantics::Symbol *
getIterationVariableSymbol(const lower::pft::Evaluation &eval);

void gatherFuncAndVarSyms(
    const ObjectList &objects, mlir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &symbolAndClause,
    bool automap = false);

int64_t getCollapseValue(const List<Clause> &clauses);

void genObjectList(const ObjectList &objects,
                   lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<mlir::Value> &operands);

void lastprivateModifierNotSupported(const omp::clause::Lastprivate &lastp,
                                     mlir::Location loc);

int64_t collectLoopRelatedInfo(
    lower::AbstractConverter &converter, mlir::Location currentLocation,
    lower::pft::Evaluation &eval, const omp::List<omp::Clause> &clauses,
    mlir::omp::LoopRelatedClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &iv);

void collectTileSizesFromOpenMPConstruct(
    const parser::OpenMPConstruct *ompCons,
    llvm::SmallVectorImpl<int64_t> &tileSizes,
    Fortran::semantics::SemanticsContext &semaCtx);

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMPUTILS_H
