// Decompose global pointer offsets before machine selection.

#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace inter {
#define GEN_PASS_DEF_DECOMPOSEWIDE
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;

namespace {

struct DecomposeWide : public inter::impl::DecomposeWideBase<DecomposeWide> {
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    SmallVector<xw::PtrAddOp> pointerAdds;
    function.walk(
        [&](xw::PtrAddOp pointerAdd) { pointerAdds.push_back(pointerAdd); });

    OpBuilder builder(function.getContext());
    for (xw::PtrAddOp pointerAdd : pointerAdds) {
      auto pointerType =
          dyn_cast<LLVM::LLVMPointerType>(pointerAdd.getBase().getType());
      if (!pointerType || pointerType.getAddressSpace() != 1 ||
          !isa<IntegerType>(pointerAdd.getOffset().getType()))
        continue;

      FailureOr<Value> offset = decompose(builder, pointerAdd.getOffset());
      if (failed(offset))
        return signalPassFailure();
      pointerAdd.getOffsetMutable().assign(*offset);
    }
  }

private:
  FailureOr<Value> emitUnsupported(Value value, const Twine &message) {
    if (Operation *operation = value.getDefiningOp())
      operation->emitOpError(message);
    else
      emitError(value.getLoc(), message);
    return failure();
  }

  void setInsertionPointAfterValue(OpBuilder &builder, Value value) {
    if (Operation *operation = value.getDefiningOp()) {
      builder.setInsertionPointAfter(operation);
      return;
    }
    builder.setInsertionPointToStart(cast<BlockArgument>(value).getOwner());
  }

  std::optional<APInt> getConstant(Value value) {
    std::optional<std::pair<APInt, bool>> constant =
        getConstantAPIntValue(OpFoldResult(value));
    if (!constant)
      return std::nullopt;
    return constant->first;
  }

  FailureOr<int64_t> getSignedConstant(Value value, StringRef diagnostic) {
    std::optional<APInt> constant = getConstant(value);
    if (!constant || constant->getBitWidth() > 64) {
      (void)emitUnsupported(value, diagnostic);
      return failure();
    }
    return constant->sextOrTrunc(64).getSExtValue();
  }

  Value createConstant(OpBuilder &builder, Location location, uint64_t value) {
    IntegerAttr attribute =
        builder.getIntegerAttr(builder.getI64Type(), APInt(64, value));
    return xw::WideConstantOp::create(builder, location, builder.getI64Type(),
                                      attribute)
        .getResult();
  }

  Value createExtend(OpBuilder &builder, Value input, bool isSigned) {
    setInsertionPointAfterValue(builder, input);
    return xw::WideExtendOp::create(builder, input.getLoc(),
                                    builder.getI64Type(), input, isSigned)
        .getResult();
  }

  LogicalResult validatePacked(Value value) {
    if (!validatedPacked.insert(value).second)
      return success();
    auto type = dyn_cast<IntegerType>(value.getType());
    if (!type || type.getWidth() != 32)
      return emitUnsupported(value, "packed pointer offset must be i32");
    if (isa<BlockArgument>(value) || getConstant(value) ||
        isa_and_nonnull<xw::GlobalIdOp, xw::LocalIdOp>(value.getDefiningOp()))
      return success();

    if (auto extension = value.getDefiningOp<LLVM::ZExtOp>())
      return validatePacked(extension.getArg());
    if (auto extension = value.getDefiningOp<LLVM::SExtOp>())
      return validatePacked(extension.getArg());
    if (auto add = value.getDefiningOp<LLVM::AddOp>()) {
      if (failed(validatePacked(add.getLhs())))
        return failure();
      return validatePacked(add.getRhs());
    }
    if (auto sub = value.getDefiningOp<LLVM::SubOp>()) {
      if (failed(validatePacked(sub.getLhs())))
        return failure();
      return validatePacked(sub.getRhs());
    }
    if (auto multiply = value.getDefiningOp<LLVM::MulOp>()) {
      Value varying = multiply.getLhs();
      std::optional<APInt> constant = getConstant(multiply.getRhs());
      if (!constant) {
        varying = multiply.getRhs();
        constant = getConstant(multiply.getLhs());
      }
      if (!constant)
        return emitUnsupported(
            value, "dynamic pointer-offset multiplication is not supported");
      if (constant->isNegative())
        return emitUnsupported(
            value, "pointer-offset multiplication requires a non-negative "
                   "constant");
      return validatePacked(varying);
    }
    if (auto shift = value.getDefiningOp<LLVM::ShlOp>()) {
      FailureOr<int64_t> amount = getSignedConstant(
          shift.getRhs(), "pointer offset requires a constant in-range shift");
      if (failed(amount) || *amount < 0 ||
          static_cast<uint64_t>(*amount) >= type.getWidth()) {
        if (succeeded(amount))
          shift.emitOpError(
              "pointer offset requires a constant in-range shift");
        return failure();
      }
      return validatePacked(shift.getLhs());
    }
    return emitUnsupported(value, "unsupported 32-bit pointer offset producer");
  }

  FailureOr<Value> decompose(OpBuilder &builder, Value value) {
    if (Value result = decomposed.lookup(value))
      return result;
    if (isa_and_nonnull<xw::WideConstantOp, xw::WideExtendOp, xw::WideAddOp,
                        xw::WideSubOp, xw::WideShlOp>(value.getDefiningOp()))
      return value;

    auto type = dyn_cast<IntegerType>(value.getType());
    if (!type)
      return emitUnsupported(value, "pointer offset must be an integer");
    if (type.getWidth() == 0 || type.getWidth() > 64)
      return emitUnsupported(value,
                             "pointer offset width must be between 1 and 64");

    if (std::optional<APInt> constant = getConstant(value)) {
      setInsertionPointAfterValue(builder, value);
      Value result = createConstant(builder, value.getLoc(),
                                    constant->sextOrTrunc(64).getZExtValue());
      decomposed.try_emplace(value, result);
      return result;
    }

    if (isa_and_nonnull<xw::GlobalIdOp, xw::LocalIdOp>(value.getDefiningOp())) {
      Value result = createExtend(builder, value, /*isSigned=*/false);
      decomposed.try_emplace(value, result);
      return result;
    }

    if (auto extension = value.getDefiningOp<LLVM::ZExtOp>())
      return decomposeExtension(builder, value, extension.getArg(),
                                /*isSigned=*/false);
    if (auto extension = value.getDefiningOp<LLVM::SExtOp>())
      return decomposeExtension(builder, value, extension.getArg(),
                                /*isSigned=*/true);

    if (type.getWidth() <= 32) {
      if (failed(validatePacked(value)))
        return failure();
      Value result = createExtend(builder, value, /*isSigned=*/true);
      decomposed.try_emplace(value, result);
      return result;
    }

    if (auto add = value.getDefiningOp<LLVM::AddOp>())
      return decomposeBinary<xw::WideAddOp>(builder, value, add.getLhs(),
                                            add.getRhs());
    if (auto sub = value.getDefiningOp<LLVM::SubOp>())
      return decomposeBinary<xw::WideSubOp>(builder, value, sub.getLhs(),
                                            sub.getRhs());
    if (auto multiply = value.getDefiningOp<LLVM::MulOp>())
      return decomposeMultiply(builder, multiply);
    if (auto shift = value.getDefiningOp<LLVM::ShlOp>())
      return decomposeShift(builder, shift);

    return emitUnsupported(value, "unsupported 64-bit pointer offset producer");
  }

  FailureOr<Value> decomposeExtension(OpBuilder &builder, Value result,
                                      Value input, bool isSigned) {
    auto inputType = dyn_cast<IntegerType>(input.getType());
    if (!inputType || inputType.getWidth() != 32)
      return emitUnsupported(
          result, "wide pointer-offset extension requires an i32 source");
    if (failed(validatePacked(input)))
      return failure();
    Value wide = createExtend(builder, input, isSigned);
    decomposed.try_emplace(result, wide);
    return wide;
  }

  template <typename OpTy>
  FailureOr<Value> decomposeBinary(OpBuilder &builder, Value original,
                                   Value lhs, Value rhs) {
    FailureOr<Value> wideLhs = decompose(builder, lhs);
    FailureOr<Value> wideRhs = decompose(builder, rhs);
    if (failed(wideLhs) || failed(wideRhs))
      return failure();
    setInsertionPointAfterValue(builder, original);
    Value result = OpTy::create(builder, original.getLoc(),
                                builder.getI64Type(), *wideLhs, *wideRhs)
                       .getResult();
    decomposed.try_emplace(original, result);
    return result;
  }

  FailureOr<Value> decomposeMultiply(OpBuilder &builder, LLVM::MulOp multiply) {
    Value varying = multiply.getLhs();
    std::optional<APInt> constant = getConstant(multiply.getRhs());
    if (!constant) {
      varying = multiply.getRhs();
      constant = getConstant(multiply.getLhs());
    }
    if (!constant)
      return emitUnsupported(
          multiply.getResult(),
          "dynamic pointer-offset multiplication is not supported");
    if (constant->isNegative())
      return emitUnsupported(
          multiply.getResult(),
          "pointer-offset multiplication requires a non-negative constant");
    if (constant->getActiveBits() > 64)
      return emitUnsupported(multiply.getResult(),
                             "pointer-offset multiplier exceeds 64 bits");

    FailureOr<Value> wide = decompose(builder, varying);
    if (failed(wide))
      return failure();
    uint64_t multiplier = constant->getZExtValue();
    setInsertionPointAfterValue(builder, multiply.getResult());
    Value result;
    for (unsigned bit = 0; bit != 64; ++bit) {
      if (!(multiplier & (uint64_t(1) << bit)))
        continue;
      Value term = *wide;
      if (bit != 0)
        term = xw::WideShlOp::create(builder, multiply.getLoc(),
                                     builder.getI64Type(), term,
                                     builder.getI64IntegerAttr(bit))
                   .getResult();
      if (result)
        result = xw::WideAddOp::create(builder, multiply.getLoc(),
                                       builder.getI64Type(), result, term)
                     .getResult();
      else
        result = term;
    }
    if (!result)
      result = createConstant(builder, multiply.getLoc(), 0);
    decomposed.try_emplace(multiply.getResult(), result);
    return result;
  }

  FailureOr<Value> decomposeShift(OpBuilder &builder, LLVM::ShlOp shift) {
    auto type = cast<IntegerType>(shift.getType());
    FailureOr<int64_t> amount = getSignedConstant(
        shift.getRhs(), "pointer offset requires a constant in-range shift");
    if (failed(amount))
      return failure();
    if (*amount < 0 || static_cast<uint64_t>(*amount) >= type.getWidth())
      return emitUnsupported(
          shift.getResult(),
          "pointer offset requires a constant in-range shift");
    FailureOr<Value> input = decompose(builder, shift.getLhs());
    if (failed(input))
      return failure();
    setInsertionPointAfterValue(builder, shift.getResult());
    Value result =
        xw::WideShlOp::create(builder, shift.getLoc(), builder.getI64Type(),
                              *input, builder.getI64IntegerAttr(*amount))
            .getResult();
    decomposed.try_emplace(shift.getResult(), result);
    return result;
  }

  DenseMap<Value, Value> decomposed;
  DenseSet<Value> validatedPacked;
};

} // namespace
