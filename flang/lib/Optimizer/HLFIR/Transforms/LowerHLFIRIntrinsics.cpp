//===- LowerHLFIRIntrinsics.cpp - Transformational intrinsics to FIR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

namespace hlfir {
#define GEN_PASS_DEF_LOWERHLFIRINTRINSICS
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

namespace {

/// Base class for passes converting transformational intrinsic operations into
/// runtime calls
template <class OP>
class HlfirIntrinsicConversion : public mlir::OpRewritePattern<OP> {
public:
  explicit HlfirIntrinsicConversion(mlir::MLIRContext *ctx)
      : mlir::OpRewritePattern<OP>{ctx} {
    // required for cases where intrinsics are chained together e.g.
    // matmul(matmul(a, b), c)
    // because converting the inner operation then invalidates the
    // outer operation: causing the pattern to apply recursively.
    //
    // This is safe because we always progress with each iteration. Circular
    // applications of operations are not expressible in MLIR because we use
    // an SSA form and one must become first. E.g.
    // %a = hlfir.matmul %b %d
    // %b = hlfir.matmul %a %d
    // cannot be written.
    // MSVC needs the this->
    this->setHasBoundedRewriteRecursion(true);
  }

protected:
  struct IntrinsicArgument {
    mlir::Value val; // allowed to be null if the argument is absent
    mlir::Type desiredType;
  };

  /// Lower the arguments to the intrinsic: adding necessary boxing and
  /// conversion to match the signature of the intrinsic in the runtime library.
  llvm::SmallVector<fir::ExtendedValue, 3>
  lowerArguments(mlir::Operation *op,
                 const llvm::ArrayRef<IntrinsicArgument> &args,
                 mlir::PatternRewriter &rewriter,
                 const fir::IntrinsicArgumentLoweringRules *argLowering) const {
    mlir::Location loc = op->getLoc();
    fir::FirOpBuilder builder{rewriter, op};

    llvm::SmallVector<fir::ExtendedValue, 3> ret;
    llvm::SmallVector<std::function<void()>, 2> cleanupFns;

    for (size_t i = 0; i < args.size(); ++i) {
      mlir::Value arg = args[i].val;
      mlir::Type desiredType = args[i].desiredType;
      if (!arg) {
        ret.emplace_back(fir::getAbsentIntrinsicArgument());
        continue;
      }
      hlfir::Entity entity{arg};

      fir::ArgLoweringRule argRules =
          fir::lowerIntrinsicArgumentAs(*argLowering, i);
      switch (argRules.lowerAs) {
      case fir::LowerIntrinsicArgAs::Value: {
        if (args[i].desiredType != arg.getType()) {
          arg = builder.createConvert(loc, desiredType, arg);
          entity = hlfir::Entity{arg};
        }
        auto [exv, cleanup] = hlfir::convertToValue(loc, builder, entity);
        if (cleanup)
          cleanupFns.push_back(*cleanup);
        ret.emplace_back(exv);
      } break;
      case fir::LowerIntrinsicArgAs::Addr: {
        auto [exv, cleanup] =
            hlfir::convertToAddress(loc, builder, entity, desiredType);
        if (cleanup)
          cleanupFns.push_back(*cleanup);
        ret.emplace_back(exv);
      } break;
      case fir::LowerIntrinsicArgAs::Box: {
        auto [box, cleanup] =
            hlfir::convertToBox(loc, builder, entity, desiredType);
        if (cleanup)
          cleanupFns.push_back(*cleanup);
        ret.emplace_back(box);
      } break;
      case fir::LowerIntrinsicArgAs::Inquired: {
        if (args[i].desiredType != arg.getType()) {
          arg = builder.createConvert(loc, desiredType, arg);
          entity = hlfir::Entity{arg};
        }
        // Place hlfir.expr in memory, and unbox fir.boxchar. Other entities
        // are translated to fir::ExtendedValue without transofrmation (notably,
        // pointers/allocatable are not dereferenced).
        // TODO: once lowering to FIR retires, UBOUND and LBOUND can be
        // simplified since the fir.box lowered here are now guarenteed to
        // contain the local lower bounds thanks to the hlfir.declare (the extra
        // rebox can be removed).
        // When taking arguments as descriptors, the runtime expect absent
        // OPTIONAL to be a nullptr to a descriptor, lowering has already
        // prepared such descriptors as needed, hence set
        // keepScalarOptionalBoxed to avoid building descriptors with a null
        // address for them.
        auto [exv, cleanup] = hlfir::translateToExtendedValue(
            loc, builder, entity, /*contiguous=*/false,
            /*keepScalarOptionalBoxed=*/true);
        if (cleanup)
          cleanupFns.push_back(*cleanup);
        ret.emplace_back(exv);
      } break;
      }
    }

    if (cleanupFns.size()) {
      auto oldInsertionPoint = builder.saveInsertionPoint();
      builder.setInsertionPointAfter(op);
      for (std::function<void()> cleanup : cleanupFns)
        cleanup();
      builder.restoreInsertionPoint(oldInsertionPoint);
    }

    return ret;
  }

  void processReturnValue(mlir::Operation *op,
                          const fir::ExtendedValue &resultExv, bool mustBeFreed,
                          fir::FirOpBuilder &builder,
                          mlir::PatternRewriter &rewriter) const {
    mlir::Location loc = op->getLoc();

    mlir::Value firBase = fir::getBase(resultExv);
    mlir::Type firBaseTy = firBase.getType();

    std::optional<hlfir::EntityWithAttributes> resultEntity;
    if (fir::isa_trivial(firBaseTy)) {
      // Some intrinsics return i1 when the original operation
      // produces fir.logical<>, so we may need to cast it.
      firBase = builder.createConvert(loc, op->getResult(0).getType(), firBase);
      resultEntity = hlfir::EntityWithAttributes{firBase};
    } else {
      resultEntity =
          hlfir::genDeclare(loc, builder, resultExv, ".tmp.intrinsic_result",
                            fir::FortranVariableFlagsAttr{});
    }

    if (resultEntity->isVariable()) {
      hlfir::AsExprOp asExpr = builder.create<hlfir::AsExprOp>(
          loc, *resultEntity, builder.createBool(loc, mustBeFreed));
      resultEntity = hlfir::EntityWithAttributes{asExpr.getResult()};
    }

    mlir::Value base = resultEntity->getBase();
    if (!mlir::isa<hlfir::ExprType>(base.getType())) {
      for (mlir::Operation *use : op->getResult(0).getUsers()) {
        if (mlir::isa<hlfir::DestroyOp>(use))
          rewriter.eraseOp(use);
      }
    }

    rewriter.replaceOp(op, base);
  }
};

// Given an integer or array of integer type, calculate the Kind parameter from
// the width for use in runtime intrinsic calls.
static unsigned getKindForType(mlir::Type ty) {
  mlir::Type eltty = hlfir::getFortranElementType(ty);
  unsigned width = mlir::cast<mlir::IntegerType>(eltty).getWidth();
  return width / 8;
}

template <class OP>
class HlfirReductionIntrinsicConversion : public HlfirIntrinsicConversion<OP> {
  using HlfirIntrinsicConversion<OP>::HlfirIntrinsicConversion;
  using IntrinsicArgument =
      typename HlfirIntrinsicConversion<OP>::IntrinsicArgument;
  using HlfirIntrinsicConversion<OP>::lowerArguments;
  using HlfirIntrinsicConversion<OP>::processReturnValue;

protected:
  auto buildNumericalArgs(OP operation, mlir::Type i32, mlir::Type logicalType,
                          mlir::PatternRewriter &rewriter,
                          std::string opName) const {
    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    inArgs.push_back({operation.getArray(), operation.getArray().getType()});
    inArgs.push_back({operation.getDim(), i32});
    inArgs.push_back({operation.getMask(), logicalType});
    auto *argLowering = fir::getIntrinsicArgumentLowering(opName);
    return lowerArguments(operation, inArgs, rewriter, argLowering);
  };

  auto buildMinMaxLocArgs(OP operation, mlir::Type i32, mlir::Type logicalType,
                          mlir::PatternRewriter &rewriter, std::string opName,
                          fir::FirOpBuilder builder) const {
    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    inArgs.push_back({operation.getArray(), operation.getArray().getType()});
    inArgs.push_back({operation.getDim(), i32});
    inArgs.push_back({operation.getMask(), logicalType});
    mlir::Value kind = builder.createIntegerConstant(
        operation->getLoc(), i32, getKindForType(operation.getType()));
    inArgs.push_back({kind, i32});
    inArgs.push_back({operation.getBack(), i32});
    auto *argLowering = fir::getIntrinsicArgumentLowering(opName);
    return lowerArguments(operation, inArgs, rewriter, argLowering);
  };

  auto buildLogicalArgs(OP operation, mlir::Type i32, mlir::Type logicalType,
                        mlir::PatternRewriter &rewriter,
                        std::string opName) const {
    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({operation.getMask(), logicalType});
    inArgs.push_back({operation.getDim(), i32});
    auto *argLowering = fir::getIntrinsicArgumentLowering(opName);
    return lowerArguments(operation, inArgs, rewriter, argLowering);
  };

public:
  llvm::LogicalResult
  matchAndRewrite(OP operation,
                  mlir::PatternRewriter &rewriter) const override {
    std::string opName;
    if constexpr (std::is_same_v<OP, hlfir::SumOp>) {
      opName = "sum";
    } else if constexpr (std::is_same_v<OP, hlfir::ProductOp>) {
      opName = "product";
    } else if constexpr (std::is_same_v<OP, hlfir::MaxvalOp>) {
      opName = "maxval";
    } else if constexpr (std::is_same_v<OP, hlfir::MinvalOp>) {
      opName = "minval";
    } else if constexpr (std::is_same_v<OP, hlfir::MinlocOp>) {
      opName = "minloc";
    } else if constexpr (std::is_same_v<OP, hlfir::MaxlocOp>) {
      opName = "maxloc";
    } else if constexpr (std::is_same_v<OP, hlfir::AnyOp>) {
      opName = "any";
    } else if constexpr (std::is_same_v<OP, hlfir::AllOp>) {
      opName = "all";
    } else {
      return mlir::failure();
    }

    fir::FirOpBuilder builder{rewriter, operation.getOperation()};
    const mlir::Location &loc = operation->getLoc();

    mlir::Type i32 = builder.getI32Type();
    mlir::Type logicalType = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());

    llvm::SmallVector<fir::ExtendedValue, 0> args;

    if constexpr (std::is_same_v<OP, hlfir::SumOp> ||
                  std::is_same_v<OP, hlfir::ProductOp> ||
                  std::is_same_v<OP, hlfir::MaxvalOp> ||
                  std::is_same_v<OP, hlfir::MinvalOp>) {
      args = buildNumericalArgs(operation, i32, logicalType, rewriter, opName);
    } else if constexpr (std::is_same_v<OP, hlfir::MinlocOp> ||
                         std::is_same_v<OP, hlfir::MaxlocOp>) {
      args = buildMinMaxLocArgs(operation, i32, logicalType, rewriter, opName,
                                builder);
    } else {
      args = buildLogicalArgs(operation, i32, logicalType, rewriter, opName);
    }

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(operation.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, opName, scalarResultType, args);

    processReturnValue(operation, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

using SumOpConversion = HlfirReductionIntrinsicConversion<hlfir::SumOp>;

using ProductOpConversion = HlfirReductionIntrinsicConversion<hlfir::ProductOp>;

using MaxvalOpConversion = HlfirReductionIntrinsicConversion<hlfir::MaxvalOp>;

using MinvalOpConversion = HlfirReductionIntrinsicConversion<hlfir::MinvalOp>;

using MinlocOpConversion = HlfirReductionIntrinsicConversion<hlfir::MinlocOp>;

using MaxlocOpConversion = HlfirReductionIntrinsicConversion<hlfir::MaxlocOp>;

using AnyOpConversion = HlfirReductionIntrinsicConversion<hlfir::AnyOp>;

using AllOpConversion = HlfirReductionIntrinsicConversion<hlfir::AllOp>;

struct CountOpConversion : public HlfirIntrinsicConversion<hlfir::CountOp> {
  using HlfirIntrinsicConversion<hlfir::CountOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CountOp count,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, count.getOperation()};
    const mlir::Location &loc = count->getLoc();

    mlir::Type i32 = builder.getI32Type();
    mlir::Type logicalType = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());

    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    inArgs.push_back({count.getMask(), logicalType});
    inArgs.push_back({count.getDim(), i32});
    mlir::Value kind = builder.createIntegerConstant(
        count->getLoc(), i32, getKindForType(count.getType()));
    inArgs.push_back({kind, i32});

    auto *argLowering = fir::getIntrinsicArgumentLowering("count");
    llvm::SmallVector<fir::ExtendedValue, 3> args =
        lowerArguments(count, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType = hlfir::getFortranElementType(count.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "count", scalarResultType, args);

    processReturnValue(count, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

struct MatmulOpConversion : public HlfirIntrinsicConversion<hlfir::MatmulOp> {
  using HlfirIntrinsicConversion<hlfir::MatmulOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::MatmulOp matmul,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, matmul.getOperation()};
    const mlir::Location &loc = matmul->getLoc();

    mlir::Value lhs = matmul.getLhs();
    mlir::Value rhs = matmul.getRhs();
    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({lhs, lhs.getType()});
    inArgs.push_back({rhs, rhs.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("matmul");
    llvm::SmallVector<fir::ExtendedValue, 2> args =
        lowerArguments(matmul, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(matmul.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "matmul", scalarResultType, args);

    processReturnValue(matmul, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

struct DotProductOpConversion
    : public HlfirIntrinsicConversion<hlfir::DotProductOp> {
  using HlfirIntrinsicConversion<hlfir::DotProductOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::DotProductOp dotProduct,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, dotProduct.getOperation()};
    const mlir::Location &loc = dotProduct->getLoc();

    mlir::Value lhs = dotProduct.getLhs();
    mlir::Value rhs = dotProduct.getRhs();
    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({lhs, lhs.getType()});
    inArgs.push_back({rhs, rhs.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("dot_product");
    llvm::SmallVector<fir::ExtendedValue, 2> args =
        lowerArguments(dotProduct, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(dotProduct.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, "dot_product", scalarResultType, args);

    processReturnValue(dotProduct, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

class TransposeOpConversion
    : public HlfirIntrinsicConversion<hlfir::TransposeOp> {
  using HlfirIntrinsicConversion<hlfir::TransposeOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::TransposeOp transpose,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, transpose.getOperation()};
    const mlir::Location &loc = transpose->getLoc();

    mlir::Value arg = transpose.getArray();
    llvm::SmallVector<IntrinsicArgument, 1> inArgs;
    inArgs.push_back({arg, arg.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("transpose");
    llvm::SmallVector<fir::ExtendedValue, 1> args =
        lowerArguments(transpose, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(transpose.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, "transpose", scalarResultType, args);

    processReturnValue(transpose, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

struct MatmulTransposeOpConversion
    : public HlfirIntrinsicConversion<hlfir::MatmulTransposeOp> {
  using HlfirIntrinsicConversion<
      hlfir::MatmulTransposeOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::MatmulTransposeOp multranspose,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, multranspose.getOperation()};
    const mlir::Location &loc = multranspose->getLoc();

    mlir::Value lhs = multranspose.getLhs();
    mlir::Value rhs = multranspose.getRhs();
    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({lhs, lhs.getType()});
    inArgs.push_back({rhs, rhs.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("matmul");
    llvm::SmallVector<fir::ExtendedValue, 2> args =
        lowerArguments(multranspose, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(multranspose.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, "matmul_transpose", scalarResultType, args);

    processReturnValue(multranspose, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

class CShiftOpConversion : public HlfirIntrinsicConversion<hlfir::CShiftOp> {
  using HlfirIntrinsicConversion<hlfir::CShiftOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CShiftOp cshift,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, cshift.getOperation()};
    const mlir::Location &loc = cshift->getLoc();

    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    mlir::Value array = cshift.getArray();
    inArgs.push_back({array, array.getType()});
    mlir::Value shift = cshift.getShift();
    inArgs.push_back({shift, shift.getType()});
    inArgs.push_back({cshift.getDim(), builder.getI32Type()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("cshift");
    llvm::SmallVector<fir::ExtendedValue, 3> args =
        lowerArguments(cshift, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(cshift.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "cshift", scalarResultType, args);

    processReturnValue(cshift, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

class ReshapeOpConversion : public HlfirIntrinsicConversion<hlfir::ReshapeOp> {
  using HlfirIntrinsicConversion<hlfir::ReshapeOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::ReshapeOp reshape,
                  mlir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, reshape.getOperation()};
    const mlir::Location &loc = reshape->getLoc();

    llvm::SmallVector<IntrinsicArgument, 4> inArgs;
    mlir::Value array = reshape.getArray();
    inArgs.push_back({array, array.getType()});
    mlir::Value shape = reshape.getShape();
    inArgs.push_back({shape, shape.getType()});
    mlir::Type noneType = builder.getNoneType();
    mlir::Value pad = reshape.getPad();
    inArgs.push_back({pad, pad ? pad.getType() : noneType});
    mlir::Value order = reshape.getOrder();
    inArgs.push_back({order, order ? order.getType() : noneType});

    auto *argLowering = fir::getIntrinsicArgumentLowering("reshape");
    llvm::SmallVector<fir::ExtendedValue, 4> args =
        lowerArguments(reshape, inArgs, rewriter, argLowering);

    mlir::Type scalarResultType =
        hlfir::getFortranElementType(reshape.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "reshape", scalarResultType, args);

    processReturnValue(reshape, resultExv, mustBeFreed, builder, rewriter);
    return mlir::success();
  }
};

class LowerHLFIRIntrinsics
    : public hlfir::impl::LowerHLFIRIntrinsicsBase<LowerHLFIRIntrinsics> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = this->getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.insert<
        MatmulOpConversion, MatmulTransposeOpConversion, AllOpConversion,
        AnyOpConversion, SumOpConversion, ProductOpConversion,
        TransposeOpConversion, CountOpConversion, DotProductOpConversion,
        MaxvalOpConversion, MinvalOpConversion, MinlocOpConversion,
        MaxlocOpConversion, CShiftOpConversion, ReshapeOpConversion>(context);

    // While conceptually this pass is performing dialect conversion, we use
    // pattern rewrites here instead of dialect conversion because this pass
    // looses array bounds from some of the expressions e.g.
    // !hlfir.expr<2xi32> -> !hlfir.expr<?xi32>
    // MLIR thinks this is a different type so dialect conversion fails.
    // Pattern rewriting only requires that the resulting IR is still valid
    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;

    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns), config))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "failure in HLFIR intrinsic lowering");
      signalPassFailure();
    }
  }
};
} // namespace
