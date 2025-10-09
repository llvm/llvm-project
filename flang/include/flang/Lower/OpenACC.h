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

/// Get a acc.private.recipe op for the given type or create it if it does not
/// exist yet.
mlir::acc::PrivateRecipeOp createOrGetPrivateRecipe(fir::FirOpBuilder &,
                                                    llvm::StringRef,
                                                    mlir::Location, mlir::Type);

/// Get a acc.reduction.recipe op for the given type or create it if it does not
/// exist yet.
mlir::acc::ReductionRecipeOp
createOrGetReductionRecipe(fir::FirOpBuilder &, llvm::StringRef, mlir::Location,
                           mlir::Type, mlir::acc::ReductionOperator,
                           llvm::SmallVector<mlir::Value> &);

/// Get a acc.firstprivate.recipe op for the given type or create it if it does
/// not exist yet.
mlir::acc::FirstprivateRecipeOp
createOrGetFirstprivateRecipe(fir::FirOpBuilder &, llvm::StringRef,
                              mlir::Location, mlir::Type,
                              llvm::SmallVector<mlir::Value> &);

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

/// Checks whether the current insertion point is inside OpenACC loop.
bool isInOpenACCLoop(fir::FirOpBuilder &);

/// Checks whether the current insertion point is inside OpenACC compute
/// construct.
bool isInsideOpenACCComputeConstruct(fir::FirOpBuilder &);

void setInsertionPointAfterOpenACCLoopIfInside(fir::FirOpBuilder &);

void genEarlyReturnInOpenACCLoop(fir::FirOpBuilder &, mlir::Location);

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
