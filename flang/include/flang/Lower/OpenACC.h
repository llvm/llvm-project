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
class StringRef;
}

namespace mlir {
class Location;
class Type;
class OpBuilder;
} // namespace mlir

namespace Fortran {
namespace parser {
struct OpenACCConstruct;
struct OpenACCDeclarativeConstruct;
} // namespace parser

namespace semantics {
class SemanticsContext;
}

namespace lower {

class AbstractConverter;

namespace pft {
struct Evaluation;
} // namespace pft

void genOpenACCConstruct(AbstractConverter &,
                         Fortran::semantics::SemanticsContext &,
                         pft::Evaluation &, const parser::OpenACCConstruct &);
void genOpenACCDeclarativeConstruct(
    AbstractConverter &, pft::Evaluation &,
    const parser::OpenACCDeclarativeConstruct &);

/// Get a acc.private.recipe op for the given type or create it if it does not
/// exist yet.
mlir::acc::PrivateRecipeOp createOrGetPrivateRecipe(mlir::OpBuilder &,
                                                    llvm::StringRef,
                                                    mlir::Location, mlir::Type);

/// Get a acc.reduction.recipe op for the given type or create it if it does not
/// exist yet.
mlir::acc::ReductionRecipeOp
createOrGetReductionRecipe(mlir::OpBuilder &, llvm::StringRef, mlir::Location,
                           mlir::Type, mlir::acc::ReductionOperator);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENACC_H
