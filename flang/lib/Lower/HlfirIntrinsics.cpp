//===-- HlfirIntrinsics.cpp -----------------------------------------------===//
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

#include "flang/Lower/HlfirIntrinsics.h"

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <mlir/IR/ValueRange.h>

namespace {

class HlfirTransformationalIntrinsic {
public:
  explicit HlfirTransformationalIntrinsic(fir::FirOpBuilder &builder,
                                          mlir::Location loc)
      : builder(builder), loc(loc) {}

  virtual ~HlfirTransformationalIntrinsic() = default;

  hlfir::EntityWithAttributes
  lower(const Fortran::lower::PreparedActualArguments &loweredActuals,
        const fir::IntrinsicArgumentLoweringRules *argLowering,
        mlir::Type stmtResultType) {
    mlir::Value res = lowerImpl(loweredActuals, argLowering, stmtResultType);
    for (const hlfir::CleanupFunction &fn : cleanupFns)
      fn();
    return {hlfir::EntityWithAttributes{res}};
  }

protected:
  fir::FirOpBuilder &builder;
  mlir::Location loc;
  llvm::SmallVector<hlfir::CleanupFunction, 3> cleanupFns;

  virtual mlir::Value
  lowerImpl(const Fortran::lower::PreparedActualArguments &loweredActuals,
            const fir::IntrinsicArgumentLoweringRules *argLowering,
            mlir::Type stmtResultType) = 0;

  llvm::SmallVector<mlir::Value> getOperandVector(
      const Fortran::lower::PreparedActualArguments &loweredActuals,
      const fir::IntrinsicArgumentLoweringRules *argLowering);

  mlir::Type computeResultType(mlir::Value argArray, mlir::Type stmtResultType);

  template <typename OP, typename... BUILD_ARGS>
  inline OP createOp(BUILD_ARGS... args) {
    return builder.create<OP>(loc, args...);
  }

  mlir::Value loadBoxAddress(
      const std::optional<Fortran::lower::PreparedActualArgument> &arg);

  void addCleanup(std::optional<hlfir::CleanupFunction> cleanup) {
    if (cleanup)
      cleanupFns.emplace_back(std::move(*cleanup));
  }
};

template <typename OP, bool HAS_MASK>
class HlfirReductionIntrinsic : public HlfirTransformationalIntrinsic {
public:
  using HlfirTransformationalIntrinsic::HlfirTransformationalIntrinsic;

protected:
  mlir::Value
  lowerImpl(const Fortran::lower::PreparedActualArguments &loweredActuals,
            const fir::IntrinsicArgumentLoweringRules *argLowering,
            mlir::Type stmtResultType) override;
};
using HlfirSumLowering = HlfirReductionIntrinsic<hlfir::SumOp, true>;
using HlfirProductLowering = HlfirReductionIntrinsic<hlfir::ProductOp, true>;
using HlfirMaxvalLowering = HlfirReductionIntrinsic<hlfir::MaxvalOp, true>;
using HlfirMinvalLowering = HlfirReductionIntrinsic<hlfir::MinvalOp, true>;
using HlfirAnyLowering = HlfirReductionIntrinsic<hlfir::AnyOp, false>;
using HlfirAllLowering = HlfirReductionIntrinsic<hlfir::AllOp, false>;

template <typename OP>
class HlfirProductIntrinsic : public HlfirTransformationalIntrinsic {
public:
  using HlfirTransformationalIntrinsic::HlfirTransformationalIntrinsic;

protected:
  mlir::Value
  lowerImpl(const Fortran::lower::PreparedActualArguments &loweredActuals,
            const fir::IntrinsicArgumentLoweringRules *argLowering,
            mlir::Type stmtResultType) override;
};
using HlfirMatmulLowering = HlfirProductIntrinsic<hlfir::MatmulOp>;
using HlfirDotProductLowering = HlfirProductIntrinsic<hlfir::DotProductOp>;

class HlfirTransposeLowering : public HlfirTransformationalIntrinsic {
public:
  using HlfirTransformationalIntrinsic::HlfirTransformationalIntrinsic;

protected:
  mlir::Value
  lowerImpl(const Fortran::lower::PreparedActualArguments &loweredActuals,
            const fir::IntrinsicArgumentLoweringRules *argLowering,
            mlir::Type stmtResultType) override;
};

class HlfirCountLowering : public HlfirTransformationalIntrinsic {
public:
  using HlfirTransformationalIntrinsic::HlfirTransformationalIntrinsic;

protected:
  mlir::Value
  lowerImpl(const Fortran::lower::PreparedActualArguments &loweredActuals,
            const fir::IntrinsicArgumentLoweringRules *argLowering,
            mlir::Type stmtResultType) override;
};

class HlfirCharExtremumLowering : public HlfirTransformationalIntrinsic {
public:
  HlfirCharExtremumLowering(fir::FirOpBuilder &builder, mlir::Location loc,
                            hlfir::CharExtremumPredicate pred)
      : HlfirTransformationalIntrinsic(builder, loc), pred{pred} {}

protected:
  mlir::Value
  lowerImpl(const Fortran::lower::PreparedActualArguments &loweredActuals,
            const fir::IntrinsicArgumentLoweringRules *argLowering,
            mlir::Type stmtResultType) override;

protected:
  hlfir::CharExtremumPredicate pred;
};

} // namespace

mlir::Value HlfirTransformationalIntrinsic::loadBoxAddress(
    const std::optional<Fortran::lower::PreparedActualArgument> &arg) {
  if (!arg)
    return mlir::Value{};

  hlfir::Entity actual = arg->getActual(loc, builder);

  if (!arg->handleDynamicOptional()) {
    if (actual.isMutableBox()) {
      // this is a box address type but is not dynamically optional. Just load
      // the box, assuming it is well formed (!fir.ref<!fir.box<...>> ->
      // !fir.box<...>)
      return builder.create<fir::LoadOp>(loc, actual.getBase());
    }
    return actual;
  }

  auto [exv, cleanup] = hlfir::translateToExtendedValue(loc, builder, actual);
  addCleanup(cleanup);

  mlir::Value isPresent = arg->getIsPresent();
  // createBox will not do create any invalid memory dereferences if exv is
  // absent. The created fir.box will not be usable, but the SelectOp below
  // ensures it won't be.
  mlir::Value box = builder.createBox(loc, exv);
  mlir::Type boxType = box.getType();
  auto absent = builder.create<fir::AbsentOp>(loc, boxType);
  auto boxOrAbsent = builder.create<mlir::arith::SelectOp>(
      loc, boxType, isPresent, box, absent);

  return boxOrAbsent;
}

llvm::SmallVector<mlir::Value> HlfirTransformationalIntrinsic::getOperandVector(
    const Fortran::lower::PreparedActualArguments &loweredActuals,
    const fir::IntrinsicArgumentLoweringRules *argLowering) {
  llvm::SmallVector<mlir::Value> operands;
  operands.reserve(loweredActuals.size());

  for (size_t i = 0; i < loweredActuals.size(); ++i) {
    std::optional<Fortran::lower::PreparedActualArgument> arg =
        loweredActuals[i];
    if (!arg) {
      operands.emplace_back();
      continue;
    }
    hlfir::Entity actual = arg->getActual(loc, builder);
    mlir::Value valArg;

    if (!argLowering) {
      valArg = hlfir::loadTrivialScalar(loc, builder, actual);
    } else {
      fir::ArgLoweringRule argRules =
          fir::lowerIntrinsicArgumentAs(*argLowering, i);
      if (argRules.lowerAs == fir::LowerIntrinsicArgAs::Box)
        valArg = loadBoxAddress(arg);
      else if (!argRules.handleDynamicOptional &&
               argRules.lowerAs != fir::LowerIntrinsicArgAs::Inquired)
        valArg = hlfir::derefPointersAndAllocatables(loc, builder, actual);
      else if (argRules.handleDynamicOptional)
        TODO(loc, "hlfir transformational intrinsic dynamically optional "
                  "argument without box lowering");
      else
        valArg = actual.getBase();
    }

    operands.emplace_back(valArg);
  }
  return operands;
}

mlir::Type
HlfirTransformationalIntrinsic::computeResultType(mlir::Value argArray,
                                                  mlir::Type stmtResultType) {
  mlir::Type normalisedResult =
      hlfir::getFortranElementOrSequenceType(stmtResultType);
  if (auto array = normalisedResult.dyn_cast<fir::SequenceType>()) {
    hlfir::ExprType::Shape resultShape =
        hlfir::ExprType::Shape{array.getShape()};
    mlir::Type elementType = array.getEleTy();
    return hlfir::ExprType::get(builder.getContext(), resultShape, elementType,
                                /*polymorphic=*/false);
  } else if (auto resCharType =
                 mlir::dyn_cast<fir::CharacterType>(stmtResultType)) {
    normalisedResult = hlfir::ExprType::get(
        builder.getContext(), hlfir::ExprType::Shape{}, resCharType, false);
  }
  return normalisedResult;
}

template <typename OP, bool HAS_MASK>
mlir::Value HlfirReductionIntrinsic<OP, HAS_MASK>::lowerImpl(
    const Fortran::lower::PreparedActualArguments &loweredActuals,
    const fir::IntrinsicArgumentLoweringRules *argLowering,
    mlir::Type stmtResultType) {
  auto operands = getOperandVector(loweredActuals, argLowering);
  mlir::Value array = operands[0];
  mlir::Value dim = operands[1];
  // dim, mask can be NULL if these arguments are not given
  if (dim)
    dim = hlfir::loadTrivialScalar(loc, builder, hlfir::Entity{dim});

  mlir::Type resultTy = computeResultType(array, stmtResultType);

  OP op;
  if constexpr (HAS_MASK)
    op = createOp<OP>(resultTy, array, dim,
                      /*mask=*/operands[2]);
  else
    op = createOp<OP>(resultTy, array, dim);
  return op;
}

template <typename OP>
mlir::Value HlfirProductIntrinsic<OP>::lowerImpl(
    const Fortran::lower::PreparedActualArguments &loweredActuals,
    const fir::IntrinsicArgumentLoweringRules *argLowering,
    mlir::Type stmtResultType) {
  auto operands = getOperandVector(loweredActuals, argLowering);
  mlir::Type resultType = computeResultType(operands[0], stmtResultType);
  return createOp<OP>(resultType, operands[0], operands[1]);
}

mlir::Value HlfirTransposeLowering::lowerImpl(
    const Fortran::lower::PreparedActualArguments &loweredActuals,
    const fir::IntrinsicArgumentLoweringRules *argLowering,
    mlir::Type stmtResultType) {
  auto operands = getOperandVector(loweredActuals, argLowering);
  hlfir::ExprType::Shape resultShape;
  mlir::Type normalisedResult =
      hlfir::getFortranElementOrSequenceType(stmtResultType);
  auto array = normalisedResult.cast<fir::SequenceType>();
  llvm::ArrayRef<int64_t> arrayShape = array.getShape();
  assert(arrayShape.size() == 2 && "arguments to transpose have a rank of 2");
  mlir::Type elementType = array.getEleTy();
  resultShape.push_back(arrayShape[0]);
  resultShape.push_back(arrayShape[1]);
  if (auto resCharType = mlir::dyn_cast<fir::CharacterType>(elementType))
    if (!resCharType.hasConstantLen()) {
      // The FunctionRef expression might have imprecise character
      // type at this point, and we can improve it by propagating
      // the constant length from the argument.
      auto argCharType = mlir::dyn_cast<fir::CharacterType>(
          hlfir::getFortranElementType(operands[0].getType()));
      if (argCharType && argCharType.hasConstantLen())
        elementType = fir::CharacterType::get(
            builder.getContext(), resCharType.getFKind(), argCharType.getLen());
    }

  mlir::Type resultTy =
      hlfir::ExprType::get(builder.getContext(), resultShape, elementType,
                           fir::isPolymorphicType(stmtResultType));
  return createOp<hlfir::TransposeOp>(resultTy, operands[0]);
}

mlir::Value HlfirCountLowering::lowerImpl(
    const Fortran::lower::PreparedActualArguments &loweredActuals,
    const fir::IntrinsicArgumentLoweringRules *argLowering,
    mlir::Type stmtResultType) {
  auto operands = getOperandVector(loweredActuals, argLowering);
  mlir::Value array = operands[0];
  mlir::Value dim = operands[1];
  if (dim)
    dim = hlfir::loadTrivialScalar(loc, builder, hlfir::Entity{dim});
  mlir::Value kind = operands[2];
  mlir::Type resultType = computeResultType(array, stmtResultType);
  return createOp<hlfir::CountOp>(resultType, array, dim, kind);
}

mlir::Value HlfirCharExtremumLowering::lowerImpl(
    const Fortran::lower::PreparedActualArguments &loweredActuals,
    const fir::IntrinsicArgumentLoweringRules *argLowering,
    mlir::Type stmtResultType) {
  auto operands = getOperandVector(loweredActuals, argLowering);
  assert(operands.size() >= 2);
  return createOp<hlfir::CharExtremumOp>(pred, mlir::ValueRange{operands});
}

std::optional<hlfir::EntityWithAttributes> Fortran::lower::lowerHlfirIntrinsic(
    fir::FirOpBuilder &builder, mlir::Location loc, const std::string &name,
    const Fortran::lower::PreparedActualArguments &loweredActuals,
    const fir::IntrinsicArgumentLoweringRules *argLowering,
    mlir::Type stmtResultType) {
  // If the result is of a derived type that may need finalization,
  // we have to use DestroyOp with 'finalize' attribute for the result
  // of the intrinsic operation.
  if (name == "sum")
    return HlfirSumLowering{builder, loc}.lower(loweredActuals, argLowering,
                                                stmtResultType);
  if (name == "product")
    return HlfirProductLowering{builder, loc}.lower(loweredActuals, argLowering,
                                                    stmtResultType);
  if (name == "any")
    return HlfirAnyLowering{builder, loc}.lower(loweredActuals, argLowering,
                                                stmtResultType);
  if (name == "all")
    return HlfirAllLowering{builder, loc}.lower(loweredActuals, argLowering,
                                                stmtResultType);
  if (name == "matmul")
    return HlfirMatmulLowering{builder, loc}.lower(loweredActuals, argLowering,
                                                   stmtResultType);
  if (name == "dot_product")
    return HlfirDotProductLowering{builder, loc}.lower(
        loweredActuals, argLowering, stmtResultType);
  // FIXME: the result may need finalization.
  if (name == "transpose")
    return HlfirTransposeLowering{builder, loc}.lower(
        loweredActuals, argLowering, stmtResultType);
  if (name == "count")
    return HlfirCountLowering{builder, loc}.lower(loweredActuals, argLowering,
                                                  stmtResultType);
  if (name == "maxval")
    return HlfirMaxvalLowering{builder, loc}.lower(loweredActuals, argLowering,
                                                   stmtResultType);
  if (name == "minval")
    return HlfirMinvalLowering{builder, loc}.lower(loweredActuals, argLowering,
                                                   stmtResultType);
  if (mlir::isa<fir::CharacterType>(stmtResultType)) {
    if (name == "min")
      return HlfirCharExtremumLowering{builder, loc,
                                       hlfir::CharExtremumPredicate::min}
          .lower(loweredActuals, argLowering, stmtResultType);
    if (name == "max")
      return HlfirCharExtremumLowering{builder, loc,
                                       hlfir::CharExtremumPredicate::max}
          .lower(loweredActuals, argLowering, stmtResultType);
  }
  return std::nullopt;
}
