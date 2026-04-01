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
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/Location.h"
#include "aiir/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include <cstdint>
#include <optional>

extern llvm::cl::opt<bool> treatIndexAsSection;

namespace fir {
class FirOpBuilder;
class RecordType;
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
  aiir::omp::DeclareTargetCaptureClause clause;
  bool automap = false;
  const semantics::Symbol &symbol;

  DeclareTargetCaptureInfo(aiir::omp::DeclareTargetCaptureClause c,
                           const semantics::Symbol &s, bool a = false)
      : clause(c), automap(a), symbol(s) {}
};

// A small helper structure for keeping track of a component members MapInfoOp
// and index data when lowering OpenMP map clauses. Keeps track of the
// placement of the component in the derived type hierarchy it rests within,
// alongside the generated aiir::omp::MapInfoOp for the mapped component.
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
  llvm::SmallVector<aiir::omp::MapInfoOp> memberMap;

  bool isDuplicateMemberMapInfo(llvm::SmallVectorImpl<int64_t> &memberIndices) {
    return llvm::find_if(memberPlacementIndices, [&](auto &memberData) {
             return llvm::equal(memberIndices, memberData);
           }) != memberPlacementIndices.end();
  }

  void addChildIndexAndMapToParent(const omp::Object &object,
                                   aiir::omp::MapInfoOp &mapOp,
                                   semantics::SemanticsContext &semaCtx);
};

void insertChildMapInfoIntoParent(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::StatementContext &stmtCtx,
    std::map<Object, OmpMapParentAndMemberData> &parentMemberIndices,
    llvm::SmallVectorImpl<aiir::Value> &mapOperands,
    llvm::SmallVectorImpl<const semantics::Symbol *> &mapSyms);

void generateMemberPlacementIndices(
    const Object &object, llvm::SmallVectorImpl<int64_t> &indices,
    Fortran::semantics::SemanticsContext &semaCtx);

bool isMemberOrParentAllocatableOrPointer(
    const Object &object, Fortran::semantics::SemanticsContext &semaCtx);

aiir::Value createParentSymAndGenIntermediateMaps(
    aiir::Location clauseLocation, Fortran::lower::AbstractConverter &converter,
    semantics::SemanticsContext &semaCtx, lower::StatementContext &stmtCtx,
    omp::ObjectList &objectList, llvm::SmallVectorImpl<int64_t> &indices,
    OmpMapParentAndMemberData &parentMemberIndices, llvm::StringRef asFortran,
    aiir::omp::ClauseMapFlags mapTypeBits);

bool requiresImplicitDefaultDeclareMapper(
    const semantics::DerivedTypeSpec &typeSpec);

omp::ObjectList gatherObjectsOf(omp::Object derivedTypeMember,
                                semantics::SemanticsContext &semaCtx);

aiir::Type getLoopVarType(lower::AbstractConverter &converter,
                          std::size_t loopVarTypeSize);

semantics::Symbol *
getIterationVariableSymbol(const lower::pft::Evaluation &eval);

void gatherFuncAndVarSyms(
    const ObjectList &objects, aiir::omp::DeclareTargetCaptureClause clause,
    llvm::SmallVectorImpl<DeclareTargetCaptureInfo> &symbolAndClause,
    bool automap = false);

int64_t getCollapseValue(const List<Clause> &clauses);

void genObjectList(const ObjectList &objects,
                   lower::AbstractConverter &converter,
                   llvm::SmallVectorImpl<aiir::Value> &operands);

void lastprivateModifierNotSupported(const omp::clause::Lastprivate &lastp,
                                     aiir::Location loc);

pft::Evaluation *getNestedDoConstruct(pft::Evaluation &eval);

int64_t collectLoopRelatedInfo(
    lower::AbstractConverter &converter, aiir::Location currentLocation,
    lower::pft::Evaluation &eval, lower::pft::Evaluation *nestedEval,
    const omp::List<omp::Clause> &clauses,
    aiir::omp::LoopRelatedClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &iv);

void collectLoopRelatedInfo(
    lower::AbstractConverter &converter, aiir::Location currentLocation,
    lower::pft::Evaluation &eval, lower::pft::Evaluation *nestedEval,
    std::int64_t collapseValue,
    // const omp::List<omp::Clause> &clauses,
    aiir::omp::LoopRelatedClauseOps &result,
    llvm::SmallVectorImpl<const semantics::Symbol *> &iv);

void collectTileSizesFromOpenMPConstruct(
    const parser::OpenMPConstruct *ompCons,
    llvm::SmallVectorImpl<int64_t> &tileSizes,
    Fortran::semantics::SemanticsContext &semaCtx);

aiir::Value genElementSizeInBytes(fir::FirOpBuilder &builder,
                                  aiir::Location loc,
                                  const aiir::DataLayout &dl,
                                  hlfir::Entity entity);

aiir::Value genAffinityAddr(Fortran::lower::AbstractConverter &converter,
                            const omp::Object &object,
                            Fortran::lower::StatementContext &stmtCtx,
                            aiir::Location loc);

aiir::Value genAffinityLen(fir::FirOpBuilder &builder, aiir::Location loc,
                           const aiir::DataLayout &dl, hlfir::Entity entity,
                           llvm::ArrayRef<aiir::Value> bounds);

struct IteratorRange {
  aiir::Value lb;
  aiir::Value ub;
  aiir::Value step;
  Fortran::semantics::Symbol *ivSym = nullptr;
};

bool hasIteratorIVReference(
    const omp::Object &object,
    const llvm::SmallPtrSetImpl<const Fortran::semantics::Symbol *> &ivSyms);

/// Default name mangler for implicit default mappers.
///
/// \param converter The converter to use for name mangling.
/// \param mapperIdName The name of the mapper to mangle.
/// \param memberName The name of the member to mangle.
void defaultMangler(Fortran::lower::AbstractConverter &converter,
                    std::string &mapperIdName, llvm::StringRef memberName);

aiir::Value genIteratorCoordinate(Fortran::lower::AbstractConverter &converter,
                                  hlfir::Entity entity,
                                  llvm::ArrayRef<aiir::Value> ivs,
                                  aiir::Location loc);

std::optional<llvm::SmallVector<aiir::Value>> getIteratorElementIndices(
    Fortran::lower::AbstractConverter &converter, const omp::Object &object,
    Fortran::lower::StatementContext &stmtCtx, aiir::Location loc);

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMPUTILS_H
