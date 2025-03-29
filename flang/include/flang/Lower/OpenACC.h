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
class Location;
class Type;
class ModuleOp;
class OpBuilder;
class Value;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace Fortran {
namespace parser {
struct AccClauseList;
struct OpenACCConstruct;
struct OpenACCDeclarativeConstruct;
struct OpenACCRoutineConstruct;
} // namespace parser

namespace semantics {
class SemanticsContext;
class Symbol;
} // namespace semantics

namespace lower {

class AbstractConverter;
class StatementContext;

namespace pft {
struct Evaluation;
} // namespace pft

using AccRoutineInfoMappingList =
    llvm::SmallVector<std::pair<std::string, mlir::SymbolRefAttr>>;

static constexpr llvm::StringRef declarePostAllocSuffix =
    "_acc_declare_update_desc_post_alloc";
static constexpr llvm::StringRef declarePreDeallocSuffix =
    "_acc_declare_update_desc_pre_dealloc";
static constexpr llvm::StringRef declarePostDeallocSuffix =
    "_acc_declare_update_desc_post_dealloc";

static constexpr llvm::StringRef privatizationRecipePrefix = "privatization";

mlir::Value genOpenACCConstruct(AbstractConverter &,
                                Fortran::semantics::SemanticsContext &,
                                pft::Evaluation &,
                                const parser::OpenACCConstruct &);
void genOpenACCDeclarativeConstruct(AbstractConverter &,
                                    Fortran::semantics::SemanticsContext &,
                                    StatementContext &,
                                    const parser::OpenACCDeclarativeConstruct &,
                                    AccRoutineInfoMappingList &);
void genOpenACCRoutineConstruct(AbstractConverter &,
                                Fortran::semantics::SemanticsContext &,
                                mlir::ModuleOp,
                                const parser::OpenACCRoutineConstruct &,
                                AccRoutineInfoMappingList &);

void finalizeOpenACCRoutineAttachment(mlir::ModuleOp,
                                      AccRoutineInfoMappingList &);

/// Get a acc.private.recipe op for the given type or create it if it does not
/// exist yet.
mlir::acc::PrivateRecipeOp createOrGetPrivateRecipe(mlir::OpBuilder &,
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
createOrGetFirstprivateRecipe(mlir::OpBuilder &, llvm::StringRef,
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

int64_t getCollapseValue(const Fortran::parser::AccClauseList &);

bool isInOpenACCLoop(fir::FirOpBuilder &);

void setInsertionPointAfterOpenACCLoopIfInside(fir::FirOpBuilder &);

void genEarlyReturnInOpenACCLoop(fir::FirOpBuilder &, mlir::Location);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENACC_H
