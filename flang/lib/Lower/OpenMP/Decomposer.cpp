//===-- Decomposer.cpp -- Compound directive decomposition ----------------===//
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

#include "Decomposer.h"

#include "Clauses.h"
#include "Utils.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Semantics/semantics.h"
#include "flang/Tools/CrossToolHelpers.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/ClauseT.h"
#include "llvm/Frontend/OpenMP/ConstructDecompositionT.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <utility>
#include <variant>

using namespace Fortran;

namespace {
using namespace Fortran::lower::omp;

struct ConstructDecomposition {
  ConstructDecomposition(mlir::ModuleOp modOp,
                         semantics::SemanticsContext &semaCtx,
                         lower::pft::Evaluation &ev,
                         llvm::omp::Directive compound,
                         const List<Clause> &clauses)
      : semaCtx(semaCtx), mod(modOp), eval(ev) {
    tomp::ConstructDecompositionT decompose(getOpenMPVersionAttribute(modOp),
                                            *this, compound,
                                            llvm::ArrayRef(clauses));
    output = std::move(decompose.output);
  }

  // Given an object, return its base object if one exists.
  std::optional<Object> getBaseObject(const Object &object) {
    return lower::omp::getBaseObject(object, semaCtx);
  }

  // Return the iteration variable of the associated loop if any.
  std::optional<Object> getLoopIterVar() {
    if (semantics::Symbol *symbol = getIterationVariableSymbol(eval))
      return Object{symbol, /*designator=*/{}};
    return std::nullopt;
  }

  semantics::SemanticsContext &semaCtx;
  mlir::ModuleOp mod;
  lower::pft::Evaluation &eval;
  List<UnitConstruct> output;
};
} // namespace

namespace Fortran::lower::omp {
LLVM_DUMP_METHOD llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                               const UnitConstruct &uc) {
  os << llvm::omp::getOpenMPDirectiveName(uc.id);
  for (auto [index, clause] : llvm::enumerate(uc.clauses)) {
    os << (index == 0 ? '\t' : ' ');
    os << llvm::omp::getOpenMPClauseName(clause.id);
  }
  return os;
}

ConstructQueue buildConstructQueue(
    mlir::ModuleOp modOp, Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval, const parser::CharBlock &source,
    llvm::omp::Directive compound, const List<Clause> &clauses) {

  ConstructDecomposition decompose(modOp, semaCtx, eval, compound, clauses);
  assert(!decompose.output.empty() && "Construct decomposition failed");

  for (UnitConstruct &uc : decompose.output) {
    assert(getLeafConstructs(uc.id).empty() && "unexpected compound directive");
    //  If some clauses are left without source information, use the directive's
    //  source.
    for (auto &clause : uc.clauses)
      if (clause.source.empty())
        clause.source = source;
  }

  return decompose.output;
}

bool matchLeafSequence(ConstructQueue::const_iterator item,
                       const ConstructQueue &queue,
                       llvm::omp::Directive directive) {
  llvm::ArrayRef<llvm::omp::Directive> leafDirs =
      llvm::omp::getLeafConstructsOrSelf(directive);

  for (auto [dir, leaf] :
       llvm::zip_longest(leafDirs, llvm::make_range(item, queue.end()))) {
    if (!dir.has_value() || !leaf.has_value())
      return false;

    if (*dir != leaf->id)
      return false;
  }

  return true;
}

bool isLastItemInQueue(ConstructQueue::const_iterator item,
                       const ConstructQueue &queue) {
  return std::next(item) == queue.end();
}
} // namespace Fortran::lower::omp
