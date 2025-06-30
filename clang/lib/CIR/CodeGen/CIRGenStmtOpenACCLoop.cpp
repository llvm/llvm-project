//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenACC Loop Stmt node as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"

#include "clang/AST/StmtOpenACC.h"

#include "mlir/Dialect/OpenACC/OpenACC.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;
using namespace mlir::acc;

void CIRGenFunction::updateLoopOpParallelism(mlir::acc::LoopOp &op,
                                             bool isOrphan,
                                             OpenACCDirectiveKind dk) {
  // Check that at least one of auto, independent, or seq is present
  // for the device-independent default clauses.
  if (op.hasParallelismFlag(mlir::acc::DeviceType::None))
    return;

  switch (dk) {
  default:
    llvm_unreachable("Invalid parent directive kind");
  case OpenACCDirectiveKind::Invalid:
  case OpenACCDirectiveKind::Parallel:
  case OpenACCDirectiveKind::ParallelLoop:
    op.addIndependent(builder.getContext(), {});
    return;
  case OpenACCDirectiveKind::Kernels:
  case OpenACCDirectiveKind::KernelsLoop:
    op.addAuto(builder.getContext(), {});
    return;
  case OpenACCDirectiveKind::Serial:
  case OpenACCDirectiveKind::SerialLoop:
    if (op.hasDefaultGangWorkerVector())
      op.addAuto(builder.getContext(), {});
    else
      op.addSeq(builder.getContext(), {});
    return;
  };
}

mlir::LogicalResult
CIRGenFunction::emitOpenACCLoopConstruct(const OpenACCLoopConstruct &s) {
  mlir::Location start = getLoc(s.getSourceRange().getBegin());
  mlir::Location end = getLoc(s.getSourceRange().getEnd());
  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto op = builder.create<LoopOp>(start, retTy, operands);

  // TODO(OpenACC): In the future we are going to need to come up with a
  // transformation here that can teach the acc.loop how to figure out the
  // 'lowerbound', 'upperbound', and 'step'.
  //
  // -'upperbound' should fortunately be pretty easy as it should be
  // in the initialization section of the cir.for loop. In Sema, we limit to
  // just the forms 'Var = init', `Type Var = init`, or `Var = init` (where it
  // is an operator= call)`.  However, as those are all necessary to emit for
  // the init section of the for loop, they should be inside the initial
  // cir.scope.
  //
  // -'upperbound' should be somewhat easy to determine. Sema is limiting this
  // to: ==, <, >, !=,  <=, >= builtin operators, the overloaded 'comparison'
  // operations, and member-call expressions.
  //
  // For the builtin comparison operators, we can pretty well deduce based on
  // the comparison what the 'end' object is going to be, and the inclusive
  // nature of it.
  //
  // For the overloaded operators, Sema will ensure that at least one side of
  // the operator is the init variable, so we can deduce the comparison there
  // too. The standard places no real bounds on WHAT the comparison operators do
  // for a `RandomAccessIterator` however, so we'll have to just 'assume' they
  // do the right thing? Note that this might be incrementing by a different
  // 'object', not an integral, so it isn't really clear to me what we can do to
  // determine the other side.
  //
  // Member-call expressions are the difficult ones. I don't think there is
  // anything we can deduce from this to determine the 'end', so we might end up
  // having to go back to Sema and make this ill-formed.
  //
  // HOWEVER: What ACC dialect REALLY cares about is the tripcount, which you
  // cannot get (in the case of `RandomAccessIterator`) from JUST 'upperbound'
  // and 'lowerbound'. We will likely have to provide a 'recipe' equivalent to
  // `std::distance` instead.  In the case of integer/pointers, it is fairly
  // simple to find: it is just the mathematical subtraction. Howver, in the
  // case of `RandomAccessIterator`, we have to enable the use of `operator-`.
  // FORTUNATELY the standard requires this to work correctly for
  // `RandomAccessIterator`, so we don't have to implement a `std::distance`
  // that loops through, like we would for a forward/etc iterator.
  //
  // 'step': Sema is currently allowing builtin ++,--, +=, -=, *=, /=, and =
  // operators. Additionally, it allows the equivalent for the operator-call, as
  // well as member-call.
  //
  // For builtin operators, we perhaps should refine the assignment here. It
  // doesn't really help us know the 'step' count at all, but we could perhaps
  // do one more step of analysis in Sema to allow something like Var = Var + 1.
  // For the others, this should get us the step reasonably well.
  //
  // For the overloaded operators, we have the same problems as for
  // 'upperbound', plus not really knowing what they do. Member-call expressions
  // are again difficult, and we might want to reconsider allowing these in
  // Sema.
  //

  // Emit all clauses.
  emitOpenACCClauses(op, s.getDirectiveKind(), s.getDirectiveLoc(),
                     s.clauses());

  updateLoopOpParallelism(op, s.isOrphanedLoopConstruct(),
                          s.getParentComputeConstructKind());

  mlir::LogicalResult stmtRes = mlir::success();
  // Emit body.
  {
    mlir::Block &block = op.getRegion().emplaceBlock();
    mlir::OpBuilder::InsertionGuard guardCase(builder);
    builder.setInsertionPointToEnd(&block);
    LexicalScope ls{*this, start, builder.getInsertionBlock()};

    stmtRes = emitStmt(s.getLoop(), /*useCurrentScope=*/true);
    builder.create<mlir::acc::YieldOp>(end);
  }

  return stmtRes;
}
