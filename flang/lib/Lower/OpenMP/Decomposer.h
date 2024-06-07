//===-- Decomposer.h -- Compound directive decomposition ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FORTRAN_LOWER_OPENMP_DECOMPOSER_H
#define FORTRAN_LOWER_OPENMP_DECOMPOSER_H

#include "Clauses.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Frontend/OpenMP/ConstructCompositionT.h"
#include "llvm/Frontend/OpenMP/ConstructDecompositionT.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class raw_ostream;
}

namespace Fortran {
namespace semantics {
class SemanticsContext;
}
namespace lower::pft {
struct Evaluation;
}
} // namespace Fortran

namespace Fortran::lower::omp {
using UnitConstruct = tomp::DirectiveWithClauses<lower::omp::Clause>;
using ConstructQueue = List<UnitConstruct>;

LLVM_DUMP_METHOD llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                               const UnitConstruct &uc);

// Given a potentially compound construct with a list of clauses that
// apply to it, break it up into individual sub-constructs each with
// the subset of applicable clauses (plus implicit clauses, if any).
// From that create a work queue where each work item corresponds to
// the sub-construct with its clauses.
ConstructQueue buildConstructQueue(mlir::ModuleOp modOp,
                                   semantics::SemanticsContext &semaCtx,
                                   lower::pft::Evaluation &eval,
                                   const parser::CharBlock &source,
                                   llvm::omp::Directive compound,
                                   const List<Clause> &clauses);

bool isLastItemInQueue(ConstructQueue::iterator item,
                       const ConstructQueue &queue);
} // namespace Fortran::lower::omp

#endif // FORTRAN_LOWER_OPENMP_DECOMPOSER_H
