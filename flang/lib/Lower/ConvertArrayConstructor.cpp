//===- ConvertArrayConstructor.cpp -- Array Constructor ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertArrayConstructor.h"
#include "flang/Evaluate/expression.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/ConvertExprToHLFIR.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

// Array constructors are lowered with three different strategies.
// All strategies are not possible with all array constructors.
//
// - Strategy 1: runtime approach (RuntimeTempStrategy).
//   This strategy works will all array constructors, but will create more
//   complex code that is harder to optimize. An allocatable temp is created,
//   it may be unallocated if the array constructor length parameters or extent
//   could not be computed. Then, the runtime is called to push lowered
//   ac-value (array constructor elements) into the allocatable. The runtime
//   will allocate or reallocate as needed while values are being pushed.
//   In the end, the allocatable contain a temporary with all the array
//   constructor evaluated elements.
//
// - Strategy 2: inlined temporary approach (InlinedTempStrategyImpl)
//   This strategy can only be used if the array constructor extent and length
//   parameters can be pre-computed without evaluating any ac-value, and if all
//   of the ac-value are scalars (at least for now).
//   A temporary is allocated inline in one go, and an index pointing at the
//   current ac-value position in the array constructor element sequence is
//   maintained and used to store ac-value as they are being lowered.
//
// - Strategy 3: "function of the indices" approach (AsElementalStrategy)
//   This strategy can only be used if the array constructor extent and length
//   parameters can be pre-computed and, if the array constructor is of the
//   form "[(scalar_expr, ac-implied-do-control)]". In this case, it is lowered
//   into an hlfir.elemental without creating any temporary in lowering. This
//   form should maximize the chance of array temporary elision when assigning
//   the array constructor, potentially reshaped, to an array variable.
//
//   The array constructor lowering looks like:
//   ```
//     strategy = selectArrayCtorLoweringStrategy(array-ctor-expr);
//     for (ac-value : array-ctor-expr)
//       if (ac-value is expression) {
//         strategy.pushValue(ac-value);
//       } else if (ac-value is implied-do) {
//         strategy.startImpliedDo(lower, upper, stride);
//         // lower nested values
//       }
//     result = strategy.finishArrayCtorLowering();
//   ```

//===----------------------------------------------------------------------===//
//   Definition of the lowering strategies. Each lowering strategy is defined
//   as a class that implements "pushValue", "startImpliedDo", and
//   "finishArrayCtorLowering".
//===----------------------------------------------------------------------===//

namespace {
/// Class that implements the "inlined temp strategy" to lower array
/// constructors. It must be further provided a CounterType class to specify how
/// the current ac-value insertion position is tracked.
template <typename CounterType>
class InlinedTempStrategyImpl {
  /// Name that will be given to the temporary allocation and hlfir.declare in
  /// the IR.
  static constexpr char tempName[] = ".tmp.arrayctor";

public:
  /// Start lowering an array constructor according to the inline strategy.
  /// The temporary is created right away.
  InlinedTempStrategyImpl(mlir::Location loc, fir::FirOpBuilder &builder,
                          fir::SequenceType declaredType, mlir::Value extent,
                          llvm::ArrayRef<mlir::Value> lengths)
      : one{builder.createIntegerConstant(loc, builder.getIndexType(), 1)},
        counter{loc, builder, one} {
    // Allocate the temporary storage.
    llvm::SmallVector<mlir::Value, 1> extents{extent};
    mlir::Value tempStorage = builder.createHeapTemporary(
        loc, declaredType, tempName, extents, lengths);
    mlir::Value shape = builder.genShape(loc, extents);
    temp =
        builder
            .create<hlfir::DeclareOp>(loc, tempStorage, tempName, shape,
                                      lengths, fir::FortranVariableFlagsAttr{})
            .getBase();
  }

  /// Push a lowered ac-value into the current insertion point and
  /// increment the insertion point.
  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 hlfir::Entity value) {
    assert(value.isScalar() && "cannot use inlined temp with array values");
    mlir::Value indexValue = counter.getAndIncrementIndex(loc, builder, one);
    hlfir::Entity tempElement = hlfir::getElementAt(
        loc, builder, hlfir::Entity{temp}, mlir::ValueRange{indexValue});
    // TODO: "copy" would probably be better than assign to ensure there are no
    // side effects (user assignments, temp, lhs finalization)?
    // This only makes a difference for derived types, so for now derived types
    // will use the runtime strategy to avoid any bad behaviors.
    builder.create<hlfir::AssignOp>(loc, value, tempElement);
  }

  /// Start a fir.do_loop with the control from an implied-do and return
  /// the loop induction variable that is the ac-do-variable value.
  /// Only usable if the counter is able to track the position through loops.
  mlir::Value startImpliedDo(mlir::Location loc, fir::FirOpBuilder &builder,
                             mlir::Value lower, mlir::Value upper,
                             mlir::Value stride) {
    if constexpr (!CounterType::canCountThroughLoops)
      fir::emitFatalError(loc, "array constructor lowering is inconsistent");
    auto loop = builder.create<fir::DoLoopOp>(loc, lower, upper, stride,
                                              /*unordered=*/false,
                                              /*finalCount=*/false);
    builder.setInsertionPointToStart(loop.getBody());
    return loop.getInductionVar();
  }

  /// Move the temporary to an hlfir.expr value (array constructors are not
  /// variables and cannot be further modified).
  hlfir::Entity finishArrayCtorLowering(mlir::Location loc,
                                        fir::FirOpBuilder &builder) {
    // Temp is created using createHeapTemporary.
    mlir::Value mustFree = builder.createBool(loc, true);
    auto hlfirExpr = builder.create<hlfir::AsExprOp>(loc, temp, mustFree);
    return hlfir::Entity{hlfirExpr};
  }

private:
  mlir::Value one;
  CounterType counter;
  mlir::Value temp;
};

/// A simple SSA value counter to lower array constructors without any
/// implied-do in the "inlined temp strategy".
/// The SSA value being tracked by the counter (hence, this
/// cannot count through loops since the SSA value in the loop becomes
/// inaccessible after the loop).
/// Semantic analysis expression rewrites unroll implied do loop with
/// compile time constant bounds (even if huge). So this minimalistic
/// counter greatly reduces the generated IR for simple but big array
/// constructors [(i,i=1,constant-expr)] that are expected to be quite
/// common.
class ValueCounter {
public:
  static constexpr bool canCountThroughLoops = false;
  ValueCounter(mlir::Location loc, fir::FirOpBuilder &builder,
               mlir::Value initialValue) {
    indexValue = initialValue;
  }

  mlir::Value getAndIncrementIndex(mlir::Location loc,
                                   fir::FirOpBuilder &builder,
                                   mlir::Value increment) {
    mlir::Value currentValue = indexValue;
    indexValue =
        builder.create<mlir::arith::AddIOp>(loc, indexValue, increment);
    return currentValue;
  }

private:
  mlir::Value indexValue;
};
using LooplessInlinedTempStrategy = InlinedTempStrategyImpl<ValueCounter>;

/// A generic memory based counter that can deal with all cases of
/// "inlined temp strategy". The counter value is stored in a temp
/// from which it is loaded, incremented, and stored every time an
/// ac-value is pushed.
class InMemoryCounter {
public:
  static constexpr bool canCountThroughLoops = true;
  InMemoryCounter(mlir::Location loc, fir::FirOpBuilder &builder,
                  mlir::Value initialValue) {
    indexVar = builder.createTemporary(loc, initialValue.getType());
    builder.create<fir::StoreOp>(loc, initialValue, indexVar);
  }

  mlir::Value getAndIncrementIndex(mlir::Location loc,
                                   fir::FirOpBuilder &builder,
                                   mlir::Value increment) const {
    mlir::Value indexValue = builder.create<fir::LoadOp>(loc, indexVar);
    indexValue =
        builder.create<mlir::arith::AddIOp>(loc, indexValue, increment);
    builder.create<fir::StoreOp>(loc, indexValue, indexVar);
    return indexValue;
  }

private:
  mlir::Value indexVar;
};
using InlinedTempStrategy = InlinedTempStrategyImpl<InMemoryCounter>;

// TODO: add and implement AsElementalStrategy.

// TODO: add and implement RuntimeTempStrategy.

/// Wrapper class that dispatch to the selected array constructor lowering
/// strategy and does nothing else.
class ArrayCtorLoweringStrategy {
public:
  template <typename A>
  ArrayCtorLoweringStrategy(A &&impl) : implVariant{std::forward<A>(impl)} {}

  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 hlfir::Entity value) {
    return std::visit(
        [&](auto &impl) { return impl.pushValue(loc, builder, value); },
        implVariant);
  }

  mlir::Value startImpliedDo(mlir::Location loc, fir::FirOpBuilder &builder,
                             mlir::Value lower, mlir::Value upper,
                             mlir::Value stride) {
    return std::visit(
        [&](auto &impl) {
          return impl.startImpliedDo(loc, builder, lower, upper, stride);
        },
        implVariant);
  }

  hlfir::Entity finishArrayCtorLowering(mlir::Location loc,
                                        fir::FirOpBuilder &builder) {
    return std::visit(
        [&](auto &impl) { return impl.finishArrayCtorLowering(loc, builder); },
        implVariant);
  }

private:
  std::variant<InlinedTempStrategy, LooplessInlinedTempStrategy> implVariant;
};
} // namespace

//===----------------------------------------------------------------------===//
//   Definition of selectArrayCtorLoweringStrategy and its helpers.
//   This is the code that analyses the evaluate::ArrayConstructor<T>,
//   pre-lowers the array constructor extent and length parameters if it can,
//   and chooses the lowering strategy.
//===----------------------------------------------------------------------===//

namespace {
/// Helper class to lower the array constructor type and its length parameters.
/// The length parameters, if any, are only lowered if this does not require
/// evaluating an ac-value.
template <typename T>
struct LengthAndTypeCollector {
  static mlir::Type collect(mlir::Location,
                            Fortran::lower::AbstractConverter &converter,
                            const Fortran::evaluate::ArrayConstructor<T> &,
                            Fortran::lower::SymMap &,
                            Fortran::lower::StatementContext &,
                            mlir::SmallVectorImpl<mlir::Value> &) {
    // Numerical and Logical types.
    return Fortran::lower::getFIRType(&converter.getMLIRContext(), T::category,
                                      T::kind, /*lenParams*/ {});
  }
};

template <>
struct LengthAndTypeCollector<Fortran::evaluate::SomeDerived> {
  static mlir::Type collect(
      mlir::Location loc, Fortran::lower::AbstractConverter &converter,
      const Fortran::evaluate::ArrayConstructor<Fortran::evaluate::SomeDerived>
          &arrayCtorExpr,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      mlir::SmallVectorImpl<mlir::Value> &lengths) {
    TODO(loc, "collect derived type and length");
  }
};

template <int Kind>
using Character =
    Fortran::evaluate::Type<Fortran::common::TypeCategory::Character, Kind>;
template <int Kind>
struct LengthAndTypeCollector<Character<Kind>> {
  static mlir::Type collect(
      mlir::Location loc, Fortran::lower::AbstractConverter &converter,
      const Fortran::evaluate::ArrayConstructor<Character<Kind>> &arrayCtorExpr,
      Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx,
      mlir::SmallVectorImpl<mlir::Value> &lengths) {
    TODO(loc, "collect character type and length");
  }
};
} // namespace

/// Does the array constructor have length parameters that
/// LengthAndTypeCollector::collect could not lower because this requires
/// lowering an ac-value and must be delayed?
static bool
failedToGatherLengthParameters(mlir::Type elementType,
                               llvm::ArrayRef<mlir::Value> lengths) {
  return (elementType.isa<fir::CharacterType>() ||
          fir::isRecordWithTypeParameters(elementType)) &&
         lengths.empty();
}

namespace {
/// Structure that analyses the ac-value and implied-do of
/// evaluate::ArrayConstructor before they are lowered. It does not generate any
/// IR. The result of this analysis pass is used to select the lowering
/// strategy.
struct ArrayCtorAnalysis {
  template <typename T>
  ArrayCtorAnalysis(
      const Fortran::evaluate::ArrayConstructor<T> &arrayCtorExpr);

  // Can the array constructor easily be rewritten into an hlfir.elemental ?
  bool isSingleImpliedDoWithOneScalarExpr() const {
    return !anyArrayExpr && isPerfectLoopNest &&
           innerNumberOfExprIfPrefectNest == 1 && depthIfPerfectLoopNest == 1;
  }

  bool anyImpliedDo{false};
  bool anyArrayExpr{false};
  bool isPerfectLoopNest{true};
  std::int64_t innerNumberOfExprIfPrefectNest = 0;
  std::int64_t depthIfPerfectLoopNest = 0;
};
} // namespace

template <typename T>
ArrayCtorAnalysis::ArrayCtorAnalysis(
    const Fortran::evaluate::ArrayConstructor<T> &arrayCtorExpr) {
  llvm::SmallVector<const Fortran::evaluate::ArrayConstructorValues<T> *>
      arrayValueListStack{&arrayCtorExpr};
  // Loop through the ac-value-list(s) of the array constructor.
  while (!arrayValueListStack.empty()) {
    std::int64_t localNumberOfImpliedDo = 0;
    std::int64_t localNumberOfExpr = 0;
    // Loop though the ac-value of an ac-value list, and add any nested
    // ac-value-list of ac-implied-do to the stack.
    for (const Fortran::evaluate::ArrayConstructorValue<T> &acValue :
         *arrayValueListStack.pop_back_val())
      std::visit(Fortran::common::visitors{
                     [&](const Fortran::evaluate::ImpliedDo<T> &impledDo) {
                       arrayValueListStack.push_back(&impledDo.values());
                       localNumberOfImpliedDo++;
                     },
                     [&](const Fortran::evaluate::Expr<T> &expr) {
                       localNumberOfExpr++;
                       anyArrayExpr = anyArrayExpr || expr.Rank() > 0;
                     }},
                 acValue.u);
    anyImpliedDo = anyImpliedDo || localNumberOfImpliedDo > 0;

    if (localNumberOfImpliedDo == 0) {
      // Leaf ac-value-list in the array constructor ac-value tree.
      if (isPerfectLoopNest)
        // This this the only leaf of the array-constructor (the array
        // constructor is a nest of single implied-do with a list of expression
        // in the last deeper implied do). e.g: "[((i+j, i=1,n)j=1,m)]".
        innerNumberOfExprIfPrefectNest = localNumberOfExpr;
    } else if (localNumberOfImpliedDo == 1 && localNumberOfExpr == 0) {
      // Perfect implied-do nest new level.
      ++depthIfPerfectLoopNest;
    } else {
      // More than one implied-do, or at least one implied-do and an expr
      // at that level. This will not form a perfect nest. Examples:
      // "[a, (i, i=1,n)]" or "[(i, i=1,n), (j, j=1,m)]".
      isPerfectLoopNest = false;
    }
  }
}

/// Helper to lower a scalar extent expression (like implied-do bounds).
static mlir::Value lowerExtentExpr(mlir::Location loc,
                                   Fortran::lower::AbstractConverter &converter,
                                   Fortran::lower::SymMap &symMap,
                                   Fortran::lower::StatementContext &stmtCtx,
                                   const Fortran::evaluate::ExtentExpr &expr) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::IndexType idxTy = builder.getIndexType();
  hlfir::Entity value = Fortran::lower::convertExprToHLFIR(
      loc, converter, toEvExpr(expr), symMap, stmtCtx);
  value = hlfir::loadTrivialScalar(loc, builder, value);
  return builder.createConvert(loc, idxTy, value);
}

/// Does \p expr contain no calls to user function?
static bool isCallFreeExpr(const Fortran::evaluate::ExtentExpr &expr) {
  for (const Fortran::semantics::Symbol &symbol :
       Fortran::evaluate::CollectSymbols(expr))
    if (Fortran::semantics::IsProcedure(symbol))
      return false;
  return true;
}

/// Core function that pre-lowers the extent and length parameters of
/// array constructors if it can, runs the ac-value analysis and
/// select the lowering strategy accordingly.
template <typename T>
static ArrayCtorLoweringStrategy selectArrayCtorLoweringStrategy(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ArrayConstructor<T> &arrayCtorExpr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::Type idxType = builder.getIndexType();
  // Try to gather the array constructor extent.
  mlir::Value extent;
  fir::SequenceType::Extent typeExtent = fir::SequenceType::getUnknownExtent();
  auto shapeExpr =
      Fortran::evaluate::GetShape(converter.getFoldingContext(), arrayCtorExpr);
  if (shapeExpr && shapeExpr->size() == 1 && (*shapeExpr)[0]) {
    const Fortran::evaluate::ExtentExpr &extentExpr = *(*shapeExpr)[0];
    if (auto constantExtent = Fortran::evaluate::ToInt64(extentExpr)) {
      typeExtent = *constantExtent;
      extent = builder.createIntegerConstant(loc, idxType, typeExtent);
    } else if (isCallFreeExpr(extentExpr)) {
      // The expression built by expression analysis for the array constructor
      // extent does not contain procedure symbols. It is side effect free.
      // This could be relaxed to allow pure procedure, but some care must
      // be taken to not bring in "unmapped" symbols from callee scopes.
      extent = lowerExtentExpr(loc, converter, symMap, stmtCtx, extentExpr);
    }
    // Otherwise, the temporary will have to be built step by step with
    // reallocation and the extent will only be known at the end of the array
    // constructor evaluation.
  }
  // Convert the array constructor type and try to gather its length parameter
  // values, if any.
  mlir::SmallVector<mlir::Value> lengths;
  mlir::Type elementType = LengthAndTypeCollector<T>::collect(
      loc, converter, arrayCtorExpr, symMap, stmtCtx, lengths);
  // Run an analysis of the array constructor ac-value.
  ArrayCtorAnalysis analysis(arrayCtorExpr);
  bool needToEvaluateOneExprToGetLengthParameters =
      failedToGatherLengthParameters(elementType, lengths);

  // Based on what was gathered and the result of the analysis, select and
  // instantiate the right lowering strategy for the array constructor.
  if (!extent || needToEvaluateOneExprToGetLengthParameters ||
      analysis.anyArrayExpr)
    TODO(loc, "Lowering of array constructor requiring the runtime");

  auto declaredType = fir::SequenceType::get({typeExtent}, elementType);
  if (analysis.isSingleImpliedDoWithOneScalarExpr())
    TODO(loc, "Lowering of array constructor as hlfir.elemental");

  if (analysis.anyImpliedDo)
    return InlinedTempStrategy(loc, builder, declaredType, extent, lengths);

  return LooplessInlinedTempStrategy(loc, builder, declaredType, extent,
                                     lengths);
}

/// Lower an ac-value expression \p expr and forward it to the selected
/// lowering strategy \p arrayBuilder,
template <typename T>
static void genAcValue(mlir::Location loc,
                       Fortran::lower::AbstractConverter &converter,
                       const Fortran::evaluate::Expr<T> &expr,
                       Fortran::lower::SymMap &symMap,
                       Fortran::lower::StatementContext &stmtCtx,
                       ArrayCtorLoweringStrategy &arrayBuilder) {
  if (expr.Rank() != 0)
    TODO(loc, "array constructor with array ac-value in HLFIR");
  // TODO: get rid of the toEvExpr indirection.
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  hlfir::Entity value = Fortran::lower::convertExprToHLFIR(
      loc, converter, toEvExpr(expr), symMap, stmtCtx);
  value = hlfir::loadTrivialScalar(loc, builder, value);
  arrayBuilder.pushValue(loc, builder, value);
}

/// Lowers an ac-value implied-do \p impledDo according to the selected
/// lowering strategy \p arrayBuilder.
template <typename T>
static void genAcValue(mlir::Location loc,
                       Fortran::lower::AbstractConverter &converter,
                       const Fortran::evaluate::ImpliedDo<T> &impledDo,
                       Fortran::lower::SymMap &symMap,
                       Fortran::lower::StatementContext &stmtCtx,
                       ArrayCtorLoweringStrategy &arrayBuilder) {
  auto lowerIndex =
      [&](const Fortran::evaluate::ExtentExpr expr) -> mlir::Value {
    return lowerExtentExpr(loc, converter, symMap, stmtCtx, expr);
  };
  mlir::Value lower = lowerIndex(impledDo.lower());
  mlir::Value upper = lowerIndex(impledDo.upper());
  mlir::Value stride = lowerIndex(impledDo.stride());
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insertPt = builder.saveInsertionPoint();
  mlir::Value impliedDoIndexValue =
      arrayBuilder.startImpliedDo(loc, builder, lower, upper, stride);
  symMap.pushImpliedDoBinding(toStringRef(impledDo.name()),
                              impliedDoIndexValue);
  stmtCtx.pushScope();

  for (const auto &acValue : impledDo.values())
    std::visit(
        [&](const auto &x) {
          genAcValue(loc, converter, x, symMap, stmtCtx, arrayBuilder);
        },
        acValue.u);

  stmtCtx.finalizeAndPop();
  symMap.popImpliedDoBinding();
  builder.restoreInsertionPoint(insertPt);
}

/// Entry point for evaluate::ArrayConstructor lowering.
template <typename T>
hlfir::EntityWithAttributes Fortran::lower::ArrayConstructorBuilder<T>::gen(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::evaluate::ArrayConstructor<T> &arrayCtorExpr,
    Fortran::lower::SymMap &symMap, Fortran::lower::StatementContext &stmtCtx) {
  fir::FirOpBuilder &builder = converter.getFirOpBuilder();
  // Select the lowering strategy given the array constructor.
  auto arrayBuilder = selectArrayCtorLoweringStrategy(
      loc, converter, arrayCtorExpr, symMap, stmtCtx);
  // Run the array lowering strategy through the ac-values.
  for (const auto &acValue : arrayCtorExpr)
    std::visit(
        [&](const auto &x) {
          genAcValue(loc, converter, x, symMap, stmtCtx, arrayBuilder);
        },
        acValue.u);
  hlfir::Entity hlfirExpr = arrayBuilder.finishArrayCtorLowering(loc, builder);
  // Insert the clean-up for the created hlfir.expr.
  fir::FirOpBuilder *bldr = &builder;
  stmtCtx.attachCleanup(
      [=]() { bldr->create<hlfir::DestroyOp>(loc, hlfirExpr); });
  return hlfir::EntityWithAttributes{hlfirExpr};
}

using namespace Fortran::evaluate;
using namespace Fortran::common;
FOR_EACH_SPECIFIC_TYPE(template class Fortran::lower::ArrayConstructorBuilder, )
