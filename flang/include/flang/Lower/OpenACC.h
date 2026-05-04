//===-- Lower/OpenACC.h -- lower OpenACC directives -------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_OPENACC_H
#define FORTRAN_LOWER_OPENACC_H

#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace llvm {
template <typename T, unsigned N>
class SmallVector;
class StringRef;
} // namespace llvm

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
class Location;
class Type;
class ModuleOp;
class OpBuilder;
class Value;
} // namespace mlir

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace Fortran {
namespace evaluate {
struct ProcedureDesignator;
} // namespace evaluate

namespace parser {
struct AccClauseList;
struct DoConstruct;
struct OpenACCConstruct;
struct OpenACCDeclarativeConstruct;
struct OpenACCRoutineConstruct;
} // namespace parser

namespace semantics {
class OpenACCRoutineInfo;
class SemanticsContext;
class Symbol;
} // namespace semantics

namespace lower {

class AbstractConverter;
class StatementContext;
class SymMap;

namespace pft {
struct Evaluation;
} // namespace pft

static constexpr llvm::StringRef declarePostAllocSuffix =
    "_acc_declare_post_alloc";
static constexpr llvm::StringRef declarePreDeallocSuffix =
    "_acc_declare_pre_dealloc";
static constexpr llvm::StringRef declarePostDeallocSuffix =
    "_acc_declare_post_dealloc";

static constexpr llvm::StringRef privatizationRecipePrefix = "privatization";

mlir::Value genOpenACCConstruct(AbstractConverter &,
                                Fortran::semantics::SemanticsContext &,
                                pft::Evaluation &,
                                const parser::OpenACCConstruct &,
                                Fortran::lower::SymMap &localSymbols);
void genOpenACCDeclarativeConstruct(
    AbstractConverter &, Fortran::semantics::SemanticsContext &,
    StatementContext &, const parser::OpenACCDeclarativeConstruct &);
void genOpenACCRoutineConstruct(
    AbstractConverter &, mlir::ModuleOp, mlir::func::FuncOp,
    const std::vector<Fortran::semantics::OpenACCRoutineInfo> &);

void attachDeclarePostAllocAction(AbstractConverter &, fir::FirOpBuilder &,
                                  const Fortran::semantics::Symbol &);
void attachDeclarePreDeallocAction(AbstractConverter &, fir::FirOpBuilder &,
                                   mlir::Value beginOpValue,
                                   const Fortran::semantics::Symbol &);
void attachDeclarePostDeallocAction(AbstractConverter &, fir::FirOpBuilder &,
                                    const Fortran::semantics::Symbol &);

void genOpenACCTerminator(fir::FirOpBuilder &, mlir::Operation *,
                          mlir::Location);

/// Used to obtain the number of contained loops to look for
/// since this is dependent on number of tile operands and collapse
/// clause.
uint64_t getLoopCountForCollapseAndTile(const Fortran::parser::AccClauseList &);

/// Parse collapse clause and return {size, force}. If absent, returns
/// {1,false}.
std::pair<uint64_t, bool>
getCollapseSizeAndForce(const Fortran::parser::AccClauseList &);

/// Checks whether the current insertion point is inside OpenACC loop.
bool isInOpenACCLoop(fir::FirOpBuilder &);

/// Record a DoConstruct as having been absorbed by a collapse clause.
/// The PFT walker should skip generating a loop for it.
void markDoConstructAsCollapsed(const Fortran::parser::DoConstruct &);

/// Check whether a DoConstruct was absorbed by a collapse clause.
bool isCollapsedDoConstruct(const Fortran::parser::DoConstruct &);

/// Clear the collapsed DoConstruct tracking set.
void clearCollapsedDoConstructs();

/// Checks whether the current insertion point is inside OpenACC compute
/// construct.
bool isInsideOpenACCComputeConstruct(fir::FirOpBuilder &);

void setInsertionPointAfterOpenACCLoopIfInside(fir::FirOpBuilder &);

void genEarlyReturnInOpenACCLoop(fir::FirOpBuilder &, mlir::Location);

/// If \p targetBlock is outside the ACC region containing the current
/// insertion point, generate the appropriate region terminator
/// (acc.terminator or acc.yield) instead of a cross-region branch.
/// Returns true if the exit was handled, false if no ACC region boundary
/// is crossed.
bool genOpenACCRegionExitBranch(fir::FirOpBuilder &, mlir::Location,
                                mlir::Block *targetBlock);

/// Generates an OpenACC loop from a do construct in order to
/// properly capture the loop bounds, parallelism determination mode,
/// and to privatize the loop variables.
/// When the conversion is rejected, nullptr is returned.
mlir::Operation *genOpenACCLoopFromDoConstruct(
    AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::SymMap &localSymbols,
    const Fortran::parser::DoConstruct &doConstruct, pft::Evaluation &eval);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENACC_H
