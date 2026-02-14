//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenMP clause nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {
template <typename OpTy>
class OpenMPClauseCIREmitter final
    : public ConstOMPClauseVisitor<OpenMPClauseCIREmitter<OpTy>> {
  OpTy &operation;
  CIRGen::CIRGenFunction &cgf;
  CIRGen::CIRGenBuilderTy &builder;

public:
  OpenMPClauseCIREmitter(OpTy &operation, CIRGen::CIRGenFunction &cgf,
                         CIRGen::CIRGenBuilderTy &builder)
      : operation(operation), cgf(cgf), builder(builder) {}

  void VisitOMPClause(const OMPClause *clause) {
    cgf.cgm.errorNYI(clause->getBeginLoc(), "OpenMPClause ",
                     clause->getClauseKind());
  }

  void emitClauses(ArrayRef<const OMPClause *> clauses) {
    for (const auto *c : clauses)
      this->Visit(c);
  }
};
template <typename OpTy>
auto makeClauseEmitter(OpTy &op, CIRGen::CIRGenFunction &cgf,
                       CIRGen::CIRGenBuilderTy &builder) {
  return OpenMPClauseCIREmitter<OpTy>(op, cgf, builder);
}
} // namespace

template <typename Op>
void CIRGenFunction::emitOpenMPClauses(Op &op,
                                       ArrayRef<const OMPClause *> clauses) {
  mlir::OpBuilder::InsertionGuard guardCase(builder);
  builder.setInsertionPoint(op);
  makeClauseEmitter(op, *this, builder).emitClauses(clauses);
}

// We're defining the template for this in a .cpp file, so we have to explicitly
// specialize the templates.
#define EXPL_SPEC(N)                                                           \
  template void CIRGenFunction::emitOpenMPClauses<N>(                          \
      N &, ArrayRef<const OMPClause *>);
EXPL_SPEC(mlir::omp::ParallelOp)
#undef EXPL_SPEC
