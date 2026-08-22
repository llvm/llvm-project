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

#include "Utils.h"
#include "flang/Lower/OpenMP/Clauses.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Parser/provenance.h"
#include "flang/Semantics/semantics.h"
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
    tomp::ConstructDecompositionT decompose(
        mlir::omp::getOpenMPVersionAttribute(modOp), *this, compound,
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
  os << llvm::omp::getOpenMPDirectiveName(uc.id, llvm::omp::FallbackVersion);
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
  // Decomposition produces no output when a clause on a compound directive
  // cannot be assigned to any of its leaf constructs. Semantics is expected to
  // have rejected such a directive already, but it does not catch every case,
  // and continuing here consumes an empty queue and reads uninitialized state.
  // Fail deterministically instead: in a release build the fall-through is
  // undefined behaviour, which shows up as an intermittent crash rather than a
  // diagnostic. See https://github.com/llvm/llvm-project/issues/211430.
  if (decompose.output.empty()) {
    mlir::Location loc = modOp.getLoc();
    parser::AllCookedSources &cooked = semaCtx.allCookedSources();
    if (std::optional<parser::ProvenanceRange> provenance =
            cooked.GetProvenanceRange(source)) {
      if (std::optional<parser::SourcePosition> pos =
              cooked.allSources().GetSourcePosition(provenance->start()))
        loc = mlir::FileLineColLoc::get(modOp.getContext(), pos->path.get(),
                                        pos->line, pos->column);
    }
    fir::emitFatalError(
        loc,
        llvm::Twine("OpenMP construct decomposition failed: a clause on '") +
            llvm::omp::getOpenMPDirectiveName(compound,
                                              llvm::omp::FallbackVersion) +
            "' cannot be applied to any of its leaf constructs",
        /*genCrashDiag=*/false);
  }

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
