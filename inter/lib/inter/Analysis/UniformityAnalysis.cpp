#include "inter/Analysis/UniformityAnalysis.h"
#include "inter/Support/Builtins.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::dataflow;

namespace inter {

static Uniformity joinOf(ArrayRef<const UniformityLattice *> operands) {
  Uniformity acc = Uniformity::bottom();
  for (const UniformityLattice *l : operands)
    acc = Uniformity::join(acc, l->getUniformity());
  return acc;
}

// Affine-preserving arithmetic: stride adds for add, scales for mul/shl by a
// constant. Bitwise ops keep the operand class when the other side is a
// constant (covers the gid & 0xffffffff truncation idiom); refine when the
// message selector needs exactness.
static Uniformity transfer(StringRef name, ArrayRef<const UniformityLattice *> ops) {
  Uniformity joined = joinOf(ops);
  if (name == LLVM::AddOp::getOperationName() ||
      name == LLVM::SubOp::getOperationName()) {
    if (ops[0]->getUniformity().kind == UniformityKind::Strided &&
        ops[1]->getUniformity().kind == UniformityKind::Strided)
      return Uniformity::strided(ops[0]->getUniformity().stride +
                                 ops[1]->getUniformity().stride);
    return joined;
  }
  if (name == LLVM::MulOp::getOperationName() ||
      name == LLVM::ShlOp::getOperationName()) {
    for (int i = 0; i < 2; ++i) {
      Uniformity a = ops[i]->getUniformity();
      Uniformity other = ops[1 - i]->getUniformity();
      if (a.kind == UniformityKind::Strided &&
          other.kind == UniformityKind::Const) {
        uint32_t shift = name == LLVM::ShlOp::getOperationName() ? 1u : 0u;
        return Uniformity::strided(a.stride * (shift ? 2u : 1u));
      }
    }
    return joined;
  }
  if (name == LLVM::AndOp::getOperationName() ||
      name == LLVM::OrOp::getOperationName() ||
      name == LLVM::XOrOp::getOperationName()) {
    for (int i = 0; i < 2; ++i) {
      Uniformity a = ops[i]->getUniformity();
      Uniformity other = ops[1 - i]->getUniformity();
      if (other.kind == UniformityKind::Const)
        return a;
    }
    return joined;
  }
  if (name == LLVM::GEPOp::getOperationName())
    return joined; // uniform base + strided index joins to Strided
  return joined;
}

LogicalResult UniformityAnalysis::visitOperation(
    Operation *op, ArrayRef<const UniformityLattice *> operands,
    ArrayRef<UniformityLattice *> results) {
  Uniformity out = Uniformity::varying();

  if (isa<LLVM::ConstantOp, arith::ConstantOp>(op)) {
    out = Uniformity::constant();
  } else if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    if (call.getCallee() && call.getCallee()->starts_with(builtins::kGetGlobalId))
      out = Uniformity::strided(1);
  } else if (isa<LLVM::LoadOp>(op)) {
    out = Uniformity::varying();
  } else if (op->getNumResults() > 0 && !operands.empty()) {
    out = transfer(op->getName().getStringRef(), operands);
  } else if (op->getNumResults() > 0) {
    out = Uniformity::varying();
  }

  for (UniformityLattice *r : results)
    propagateIfChanged(r, r->joinUniformity(out));
  return success();
}

void UniformityAnalysis::setToEntryState(UniformityLattice *lattice) {
  Value anchor = lattice->getAnchor();
  // Kernel arguments are workgroup-uniform by definition (cross-thread
  // payload). Everything else enters at the pessimistic state.
  if (auto barg = dyn_cast<BlockArgument>(anchor)) {
    if (barg.getOwner()->isEntryBlock() &&
        isa<func::FuncOp>(barg.getOwner()->getParentOp())) {
      lattice->setUniformity(Uniformity::uniform());
      return;
    }
  }
  lattice->setUniformity(Uniformity::varying());
}

} // namespace inter
