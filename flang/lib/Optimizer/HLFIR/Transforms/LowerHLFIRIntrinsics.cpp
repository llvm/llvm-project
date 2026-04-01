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
#include "aiir/IR/BuiltinDialect.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Transforms/GreedyPatternRewriteDriver.h"
#include <optional>

namespace hlfir {
#define GEN_PASS_DEF_LOWERHLFIRINTRINSICS
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

namespace {

/// Base class for passes converting transformational intrinsic operations into
/// runtime calls
template <class OP>
class HlfirIntrinsicConversion : public aiir::OpRewritePattern<OP> {
public:
  explicit HlfirIntrinsicConversion(aiir::AIIRContext *ctx)
      : aiir::OpRewritePattern<OP>{ctx} {
    // required for cases where intrinsics are chained together e.g.
    // matmul(matmul(a, b), c)
    // because converting the inner operation then invalidates the
    // outer operation: causing the pattern to apply recursively.
    //
    // This is safe because we always progress with each iteration. Circular
    // applications of operations are not expressible in AIIR because we use
    // an SSA form and one must become first. E.g.
    // %a = hlfir.matmul %b %d
    // %b = hlfir.matmul %a %d
    // cannot be written.
    // MSVC needs the this->
    this->setHasBoundedRewriteRecursion(true);
  }

protected:
  struct IntrinsicArgument {
    aiir::Value val; // allowed to be null if the argument is absent
    aiir::Type desiredType;
  };

  /// Lower the arguments to the intrinsic: adding necessary boxing and
  /// conversion to match the signature of the intrinsic in the runtime library.
  llvm::SmallVector<fir::ExtendedValue, 3>
  lowerArguments(aiir::Operation *op,
                 const llvm::ArrayRef<IntrinsicArgument> &args,
                 aiir::PatternRewriter &rewriter,
                 const fir::IntrinsicArgumentLoweringRules *argLowering) const {
    aiir::Location loc = op->getLoc();
    fir::FirOpBuilder builder{rewriter, op};

    llvm::SmallVector<fir::ExtendedValue, 3> ret;
    llvm::SmallVector<std::function<void()>, 2> cleanupFns;

    for (size_t i = 0; i < args.size(); ++i) {
      aiir::Value arg = args[i].val;
      aiir::Type desiredType = args[i].desiredType;
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

  void processReturnValue(aiir::Operation *op,
                          const fir::ExtendedValue &resultExv, bool mustBeFreed,
                          fir::FirOpBuilder &builder,
                          aiir::PatternRewriter &rewriter) const {
    aiir::Location loc = op->getLoc();

    aiir::Value firBase = fir::getBase(resultExv);
    aiir::Type firBaseTy = firBase.getType();

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
      hlfir::AsExprOp asExpr = hlfir::AsExprOp::create(
          builder, loc, *resultEntity, builder.createBool(loc, mustBeFreed));
      resultEntity = hlfir::EntityWithAttributes{asExpr.getResult()};
    }

    aiir::Value base = resultEntity->getBase();
    if (!aiir::isa<hlfir::ExprType>(base.getType())) {
      for (aiir::Operation *use : op->getResult(0).getUsers()) {
        if (aiir::isa<hlfir::DestroyOp>(use))
          rewriter.eraseOp(use);
      }
    }

    rewriter.replaceOp(op, base);
  }
};

// Given an integer or array of integer type, calculate the Kind parameter from
// the width for use in runtime intrinsic calls.
static unsigned getKindForType(aiir::Type ty) {
  aiir::Type eltty = hlfir::getFortranElementType(ty);
  unsigned width = aiir::cast<aiir::IntegerType>(eltty).getWidth();
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
  auto buildNumericalArgs(OP operation, aiir::Type i32, aiir::Type logicalType,
                          aiir::PatternRewriter &rewriter,
                          std::string opName) const {
    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    inArgs.push_back({operation.getArray(), operation.getArray().getType()});
    inArgs.push_back({operation.getDim(), i32});
    inArgs.push_back({operation.getMask(), logicalType});
    auto *argLowering = fir::getIntrinsicArgumentLowering(opName);
    return lowerArguments(operation, inArgs, rewriter, argLowering);
  };

  auto buildMinMaxLocArgs(OP operation, aiir::Type i32, aiir::Type logicalType,
                          aiir::PatternRewriter &rewriter, std::string opName,
                          fir::FirOpBuilder builder) const {
    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    inArgs.push_back({operation.getArray(), operation.getArray().getType()});
    inArgs.push_back({operation.getDim(), i32});
    inArgs.push_back({operation.getMask(), logicalType});
    aiir::Value kind = builder.createIntegerConstant(
        operation->getLoc(), i32, getKindForType(operation.getType()));
    inArgs.push_back({kind, i32});
    inArgs.push_back({operation.getBack(), i32});
    auto *argLowering = fir::getIntrinsicArgumentLowering(opName);
    return lowerArguments(operation, inArgs, rewriter, argLowering);
  };

  auto buildLogicalArgs(OP operation, aiir::Type i32, aiir::Type logicalType,
                        aiir::PatternRewriter &rewriter,
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
                  aiir::PatternRewriter &rewriter) const override {
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
      return aiir::failure();
    }

    fir::FirOpBuilder builder{rewriter, operation.getOperation()};
    const aiir::Location &loc = operation->getLoc();

    aiir::Type i32 = builder.getI32Type();
    aiir::Type logicalType = fir::LogicalType::get(
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

    aiir::Type scalarResultType =
        hlfir::getFortranElementType(operation.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, opName, scalarResultType, args);

    processReturnValue(operation, resultExv, mustBeFreed, builder, rewriter);
    return aiir::success();
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
                  aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, count.getOperation()};
    const aiir::Location &loc = count->getLoc();

    aiir::Type i32 = builder.getI32Type();
    aiir::Type logicalType = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());

    llvm::SmallVector<IntrinsicArgument, 3> inArgs;
    inArgs.push_back({count.getMask(), logicalType});
    inArgs.push_back({count.getDim(), i32});
    aiir::Value kind = builder.createIntegerConstant(
        count->getLoc(), i32, getKindForType(count.getType()));
    inArgs.push_back({kind, i32});

    auto *argLowering = fir::getIntrinsicArgumentLowering("count");
    llvm::SmallVector<fir::ExtendedValue, 3> args =
        lowerArguments(count, inArgs, rewriter, argLowering);

    aiir::Type scalarResultType = hlfir::getFortranElementType(count.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "count", scalarResultType, args);

    processReturnValue(count, resultExv, mustBeFreed, builder, rewriter);
    return aiir::success();
  }
};

struct MatmulOpConversion : public HlfirIntrinsicConversion<hlfir::MatmulOp> {
  using HlfirIntrinsicConversion<hlfir::MatmulOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::MatmulOp matmul,
                  aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, matmul.getOperation()};
    const aiir::Location &loc = matmul->getLoc();

    aiir::Value lhs = matmul.getLhs();
    aiir::Value rhs = matmul.getRhs();
    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({lhs, lhs.getType()});
    inArgs.push_back({rhs, rhs.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("matmul");
    llvm::SmallVector<fir::ExtendedValue, 2> args =
        lowerArguments(matmul, inArgs, rewriter, argLowering);

    aiir::Type scalarResultType =
        hlfir::getFortranElementType(matmul.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "matmul", scalarResultType, args);

    processReturnValue(matmul, resultExv, mustBeFreed, builder, rewriter);
    return aiir::success();
  }
};

struct DotProductOpConversion
    : public HlfirIntrinsicConversion<hlfir::DotProductOp> {
  using HlfirIntrinsicConversion<hlfir::DotProductOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::DotProductOp dotProduct,
                  aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, dotProduct.getOperation()};
    const aiir::Location &loc = dotProduct->getLoc();

    aiir::Value lhs = dotProduct.getLhs();
    aiir::Value rhs = dotProduct.getRhs();
    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({lhs, lhs.getType()});
    inArgs.push_back({rhs, rhs.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("dot_product");
    llvm::SmallVector<fir::ExtendedValue, 2> args =
        lowerArguments(dotProduct, inArgs, rewriter, argLowering);

    aiir::Type scalarResultType =
        hlfir::getFortranElementType(dotProduct.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, "dot_product", scalarResultType, args);

    processReturnValue(dotProduct, resultExv, mustBeFreed, builder, rewriter);
    return aiir::success();
  }
};

class TransposeOpConversion
    : public HlfirIntrinsicConversion<hlfir::TransposeOp> {
  using HlfirIntrinsicConversion<hlfir::TransposeOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::TransposeOp transpose,
                  aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, transpose.getOperation()};
    const aiir::Location &loc = transpose->getLoc();

    aiir::Value arg = transpose.getArray();
    llvm::SmallVector<IntrinsicArgument, 1> inArgs;
    inArgs.push_back({arg, arg.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("transpose");
    llvm::SmallVector<fir::ExtendedValue, 1> args =
        lowerArguments(transpose, inArgs, rewriter, argLowering);

    aiir::Type scalarResultType =
        hlfir::getFortranElementType(transpose.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, "transpose", scalarResultType, args);

    processReturnValue(transpose, resultExv, mustBeFreed, builder, rewriter);
    return aiir::success();
  }
};

struct MatmulTransposeOpConversion
    : public HlfirIntrinsicConversion<hlfir::MatmulTransposeOp> {
  using HlfirIntrinsicConversion<
      hlfir::MatmulTransposeOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::MatmulTransposeOp multranspose,
                  aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, multranspose.getOperation()};
    const aiir::Location &loc = multranspose->getLoc();

    aiir::Value lhs = multranspose.getLhs();
    aiir::Value rhs = multranspose.getRhs();
    llvm::SmallVector<IntrinsicArgument, 2> inArgs;
    inArgs.push_back({lhs, lhs.getType()});
    inArgs.push_back({rhs, rhs.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("matmul");
    llvm::SmallVector<fir::ExtendedValue, 2> args =
        lowerArguments(multranspose, inArgs, rewriter, argLowering);

    aiir::Type scalarResultType =
        hlfir::getFortranElementType(multranspose.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, "matmul_transpose", scalarResultType, args);

    processReturnValue(multranspose, resultExv, mustBeFreed, builder, rewriter);
    return aiir::success();
  }
};

// A converter for hlfir.cshift and hlfir.eoshift.
template <typename T>
class ArrayShiftOpConversion : public HlfirIntrinsicConversion<T> {
  using HlfirIntrinsicConversion<T>::HlfirIntrinsicConversion;
  using HlfirIntrinsicConversion<T>::lowerArguments;
  using HlfirIntrinsicConversion<T>::processReturnValue;
  using typename HlfirIntrinsicConversion<T>::IntrinsicArgument;

  llvm::LogicalResult
  matchAndRewrite(T op, aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, op.getOperation()};
    const aiir::Location &loc = op->getLoc();

    llvm::SmallVector<IntrinsicArgument, 4> inArgs;
    llvm::StringRef intrinsicName{[]() {
      if constexpr (std::is_same_v<T, hlfir::EOShiftOp>)
        return "eoshift";
      else if constexpr (std::is_same_v<T, hlfir::CShiftOp>)
        return "cshift";
      else
        llvm_unreachable("unsupported array shift");
    }()};

    aiir::Value array = op.getArray();
    inArgs.push_back({array, array.getType()});
    aiir::Value shift = op.getShift();
    inArgs.push_back({shift, shift.getType()});
    if constexpr (std::is_same_v<T, hlfir::EOShiftOp>) {
      aiir::Value boundary = op.getBoundary();
      inArgs.push_back({boundary, boundary ? boundary.getType() : nullptr});
    }
    inArgs.push_back({op.getDim(), builder.getI32Type()});

    auto *argLowering = fir::getIntrinsicArgumentLowering(intrinsicName);
    llvm::SmallVector<fir::ExtendedValue, 3> args =
        lowerArguments(op, inArgs, rewriter, argLowering);

    aiir::Type scalarResultType = hlfir::getFortranElementType(op.getType());

    auto [resultExv, mustBeFreed] = fir::genIntrinsicCall(
        builder, loc, intrinsicName, scalarResultType, args);

    processReturnValue(op, resultExv, mustBeFreed, builder, rewriter);
    return aiir::success();
  }
};

class ReshapeOpConversion : public HlfirIntrinsicConversion<hlfir::ReshapeOp> {
  using HlfirIntrinsicConversion<hlfir::ReshapeOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::ReshapeOp reshape,
                  aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, reshape.getOperation()};
    const aiir::Location &loc = reshape->getLoc();

    llvm::SmallVector<IntrinsicArgument, 4> inArgs;
    aiir::Value array = reshape.getArray();
    inArgs.push_back({array, array.getType()});
    aiir::Value shape = reshape.getShape();
    inArgs.push_back({shape, shape.getType()});
    aiir::Type noneType = builder.getNoneType();
    aiir::Value pad = reshape.getPad();
    inArgs.push_back({pad, pad ? pad.getType() : noneType});
    aiir::Value order = reshape.getOrder();
    inArgs.push_back({order, order ? order.getType() : noneType});

    auto *argLowering = fir::getIntrinsicArgumentLowering("reshape");
    llvm::SmallVector<fir::ExtendedValue, 4> args =
        lowerArguments(reshape, inArgs, rewriter, argLowering);

    aiir::Type scalarResultType =
        hlfir::getFortranElementType(reshape.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "reshape", scalarResultType, args);

    processReturnValue(reshape, resultExv, mustBeFreed, builder, rewriter);
    return aiir::success();
  }
};

class CmpCharOpConversion : public HlfirIntrinsicConversion<hlfir::CmpCharOp> {
  using HlfirIntrinsicConversion<hlfir::CmpCharOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CmpCharOp cmp,
                  aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, cmp.getOperation()};
    const aiir::Location &loc = cmp->getLoc();
    hlfir::Entity lhs{cmp.getLchr()};
    hlfir::Entity rhs{cmp.getRchr()};

    auto [lhsExv, lhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, lhs);
    auto [rhsExv, rhsCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, rhs);

    auto resultVal = fir::runtime::genCharCompare(
        builder, loc, cmp.getPredicate(), lhsExv, rhsExv);
    if (lhsCleanUp || rhsCleanUp) {
      aiir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(cmp);
      if (lhsCleanUp)
        (*lhsCleanUp)();
      if (rhsCleanUp)
        (*rhsCleanUp)();
    }
    auto resultEntity = hlfir::EntityWithAttributes{resultVal};

    processReturnValue(cmp, resultEntity, /*mustBeFreed=*/false, builder,
                       rewriter);
    return aiir::success();
  }
};

class CharTrimOpConversion
    : public HlfirIntrinsicConversion<hlfir::CharTrimOp> {
  using HlfirIntrinsicConversion<hlfir::CharTrimOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::CharTrimOp trim,
                  aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, trim.getOperation()};
    const aiir::Location &loc = trim->getLoc();

    llvm::SmallVector<IntrinsicArgument, 1> inArgs;
    aiir::Value chr = trim.getChr();
    inArgs.push_back({chr, chr.getType()});

    auto *argLowering = fir::getIntrinsicArgumentLowering("trim");
    llvm::SmallVector<fir::ExtendedValue, 1> args =
        lowerArguments(trim, inArgs, rewriter, argLowering);

    aiir::Type resultType = hlfir::getFortranElementType(trim.getType());

    auto [resultExv, mustBeFreed] =
        fir::genIntrinsicCall(builder, loc, "trim", resultType, args);

    processReturnValue(trim, resultExv, mustBeFreed, builder, rewriter);
    return aiir::success();
  }
};

class IndexOpConversion : public HlfirIntrinsicConversion<hlfir::IndexOp> {
  using HlfirIntrinsicConversion<hlfir::IndexOp>::HlfirIntrinsicConversion;

  llvm::LogicalResult
  matchAndRewrite(hlfir::IndexOp op,
                  aiir::PatternRewriter &rewriter) const override {
    fir::FirOpBuilder builder{rewriter, op.getOperation()};
    const aiir::Location &loc = op->getLoc();
    hlfir::Entity substr{op.getSubstr()};
    hlfir::Entity str{op.getStr()};

    auto [substrExv, substrCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, substr);
    auto [strExv, strCleanUp] =
        hlfir::translateToExtendedValue(loc, builder, str);

    aiir::Value back = op.getBack();
    if (!back)
      back = builder.createBool(loc, false);

    aiir::Value result =
        fir::runtime::genIndex(builder, loc, strExv, substrExv, back);
    result = builder.createConvert(loc, op.getType(), result);
    if (strCleanUp || substrCleanUp) {
      aiir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(op);
      if (strCleanUp)
        (*strCleanUp)();
      if (substrCleanUp)
        (*substrCleanUp)();
    }
    auto resultEntity = hlfir::EntityWithAttributes{result};

    processReturnValue(op, resultEntity, /*mustBeFreed=*/false, builder,
                       rewriter);
    return aiir::success();
  }
};

class LowerHLFIRIntrinsics
    : public hlfir::impl::LowerHLFIRIntrinsicsBase<LowerHLFIRIntrinsics> {
public:
  void runOnOperation() override {
    aiir::ModuleOp module = this->getOperation();
    aiir::AIIRContext *context = &getContext();
    aiir::RewritePatternSet patterns(context);
    patterns.insert<
        MatmulOpConversion, MatmulTransposeOpConversion, AllOpConversion,
        AnyOpConversion, SumOpConversion, ProductOpConversion,
        TransposeOpConversion, CountOpConversion, DotProductOpConversion,
        MaxvalOpConversion, MinvalOpConversion, MinlocOpConversion,
        MaxlocOpConversion, ArrayShiftOpConversion<hlfir::CShiftOp>,
        ArrayShiftOpConversion<hlfir::EOShiftOp>, ReshapeOpConversion,
        CmpCharOpConversion, CharTrimOpConversion, IndexOpConversion>(context);

    // While conceptually this pass is performing dialect conversion, we use
    // pattern rewrites here instead of dialect conversion because this pass
    // looses array bounds from some of the expressions e.g.
    // !hlfir.expr<2xi32> -> !hlfir.expr<?xi32>
    // AIIR thinks this is a different type so dialect conversion fails.
    // Pattern rewriting only requires that the resulting IR is still valid
    aiir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks
    config.setRegionSimplificationLevel(
        aiir::GreedySimplifyRegionLevel::Disabled);

    if (aiir::failed(
            aiir::applyPatternsGreedily(module, std::move(patterns), config))) {
      aiir::emitError(aiir::UnknownLoc::get(context),
                      "failure in HLFIR intrinsic lowering");
      signalPassFailure();
    }
  }
};
} // namespace
