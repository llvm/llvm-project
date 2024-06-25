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
#include "flang/Optimizer/Builder/Runtime/ArrayConstructor.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/TemporaryStorage.h"
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
//         strategy.startImpliedDoScope();
//         // lower nested values
//         ...
//         strategy.endImpliedDoScope();
//       }
//     result = strategy.finishArrayCtorLowering();
//   ```

//===----------------------------------------------------------------------===//
//   Definition of the lowering strategies. Each lowering strategy is defined
//   as a class that implements "pushValue", "startImpliedDo" and
//   "finishArrayCtorLowering". A strategy may optionally override
//   "startImpliedDoScope" and "endImpliedDoScope" virtual methods
//   of its base class StrategyBase.
//===----------------------------------------------------------------------===//

namespace {
/// Class provides common implementation of scope push/pop methods
/// that update StatementContext scopes and SymMap bindings.
/// They might be overridden by the lowering strategies, e.g.
/// see AsElementalStrategy.
class StrategyBase {
public:
  StrategyBase(Fortran::lower::StatementContext &stmtCtx,
               Fortran::lower::SymMap &symMap)
      : stmtCtx{stmtCtx}, symMap{symMap} {};
  virtual ~StrategyBase() = default;

  virtual void startImpliedDoScope(llvm::StringRef doName,
                                   mlir::Value indexValue) {
    symMap.pushImpliedDoBinding(doName, indexValue);
    stmtCtx.pushScope();
  }

  virtual void endImpliedDoScope() {
    stmtCtx.finalizeAndPop();
    symMap.popImpliedDoBinding();
  }

protected:
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
};

/// Class that implements the "inlined temp strategy" to lower array
/// constructors. It must be provided a boolean to indicate if the array
/// constructor has any implied-do-loop.
template <bool hasLoops>
class InlinedTempStrategyImpl : public StrategyBase,
                                public fir::factory::HomogeneousScalarStack {
  /// Name that will be given to the temporary allocation and hlfir.declare in
  /// the IR.
  static constexpr char tempName[] = ".tmp.arrayctor";

public:
  /// Start lowering an array constructor according to the inline strategy.
  /// The temporary is created right away.
  InlinedTempStrategyImpl(mlir::Location loc, fir::FirOpBuilder &builder,
                          Fortran::lower::StatementContext &stmtCtx,
                          Fortran::lower::SymMap &symMap,
                          fir::SequenceType declaredType, mlir::Value extent,
                          llvm::ArrayRef<mlir::Value> lengths)
      : StrategyBase{stmtCtx, symMap},
        fir::factory::HomogeneousScalarStack{
            loc,      builder, declaredType,
            extent,   lengths, /*allocateOnHeap=*/true,
            hasLoops, tempName} {}

  /// Push a lowered ac-value into the current insertion point and
  /// increment the insertion point.
  using fir::factory::HomogeneousScalarStack::pushValue;

  /// Start a fir.do_loop with the control from an implied-do and return
  /// the loop induction variable that is the ac-do-variable value.
  /// Only usable if the counter is able to track the position through loops.
  mlir::Value startImpliedDo(mlir::Location loc, fir::FirOpBuilder &builder,
                             mlir::Value lower, mlir::Value upper,
                             mlir::Value stride) {
    if constexpr (!hasLoops)
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
    return moveStackAsArrayExpr(loc, builder);
  }
};

/// Semantic analysis expression rewrites unroll implied do loop with
/// compile time constant bounds (even if huge). So using a minimalistic
/// counter greatly reduces the generated IR for simple but big array
/// constructors [(i,i=1,constant-expr)] that are expected to be quite
/// common.
using LooplessInlinedTempStrategy = InlinedTempStrategyImpl</*hasLoops=*/false>;
/// A generic memory based counter that can deal with all cases of
/// "inlined temp strategy". The counter value is stored in a temp
/// from which it is loaded, incremented, and stored every time an
/// ac-value is pushed.
using InlinedTempStrategy = InlinedTempStrategyImpl</*hasLoops=*/true>;

/// Class that implements the "as function of the indices" lowering strategy.
/// It will lower [(scalar_expr(i), i=l,u,s)] to:
/// ```
///   %extent = max((%u-%l+1)/%s, 0)
///   %shape = fir.shape %extent
///   %elem = hlfir.elemental %shape {
///     ^bb0(%pos:index):
///      %i = %l+(%i-1)*%s
///      %value = scalar_expr(%i)
///       hlfir.yield_element %value
///    }
/// ```
/// That way, no temporary is created in lowering, and if the array constructor
/// is part of a more complex elemental expression, or an assignment, it will be
/// trivial to "inline" it in the expression or assignment loops if allowed by
/// alias analysis.
/// This lowering is however only possible for the form of array constructors as
/// in the illustration above. It could be extended to deeper independent
/// implied-do nest and wrapped in an hlfir.reshape to a rank 1 array. But this
/// op does not exist yet, so this is left for the future if it appears
/// profitable.
class AsElementalStrategy : public StrategyBase {
public:
  /// The constructor only gathers the operands to create the hlfir.elemental.
  AsElementalStrategy(mlir::Location loc, fir::FirOpBuilder &builder,
                      Fortran::lower::StatementContext &stmtCtx,
                      Fortran::lower::SymMap &symMap,
                      fir::SequenceType declaredType, mlir::Value extent,
                      llvm::ArrayRef<mlir::Value> lengths)
      : StrategyBase{stmtCtx, symMap}, shape{builder.genShape(loc, {extent})},
        lengthParams{lengths.begin(), lengths.end()},
        exprType{getExprType(declaredType)} {}

  static hlfir::ExprType getExprType(fir::SequenceType declaredType) {
    // Note: 7.8 point 4: the dynamic type of an array constructor is its static
    // type, it is not polymorphic.
    return hlfir::ExprType::get(declaredType.getContext(),
                                declaredType.getShape(),
                                declaredType.getEleTy(),
                                /*isPolymorphic=*/false);
  }

  /// Create the hlfir.elemental and compute the ac-implied-do-index value
  /// given the lower bound and stride (compute "%i" in the illustration above).
  mlir::Value startImpliedDo(mlir::Location loc, fir::FirOpBuilder &builder,
                             mlir::Value lower, mlir::Value upper,
                             mlir::Value stride) {
    assert(!elementalOp && "expected only one implied-do");
    mlir::Value one =
        builder.createIntegerConstant(loc, builder.getIndexType(), 1);
    elementalOp = builder.create<hlfir::ElementalOp>(
        loc, exprType, shape,
        /*mold=*/nullptr, lengthParams, /*isUnordered=*/true);
    builder.setInsertionPointToStart(elementalOp.getBody());
    // implied-do-index = lower+((i-1)*stride)
    mlir::Value diff = builder.create<mlir::arith::SubIOp>(
        loc, elementalOp.getIndices()[0], one);
    mlir::Value mul = builder.create<mlir::arith::MulIOp>(loc, diff, stride);
    mlir::Value add = builder.create<mlir::arith::AddIOp>(loc, lower, mul);
    return add;
  }

  /// Create the elemental hlfir.yield_element with the scalar ac-value.
  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 hlfir::Entity value) {
    assert(value.isScalar() && "cannot use hlfir.elemental with array values");
    assert(elementalOp && "array constructor must contain an outer implied-do");
    mlir::Value elementResult = value;
    if (fir::isa_trivial(elementResult.getType()))
      elementResult =
          builder.createConvert(loc, exprType.getElementType(), elementResult);

    // The clean-ups associated with the implied-do body operations
    // must be initiated before the YieldElementOp, so we have to pop the scope
    // right now.
    stmtCtx.finalizeAndPop();

    // This is a hacky way to get rid of the DestroyOp clean-up
    // associated with the final ac-value result if it is hlfir.expr.
    // Example:
    //   ... = (/(REPEAT(REPEAT(CHAR(i),2),2),i=1,n)/)
    // Each intrinsic call lowering will produce hlfir.expr result
    // with the associated clean-up, but only the last of them
    // is wrong. It is wrong because the value is used in hlfir.yield_element,
    // so it cannot be destroyed.
    mlir::Operation *destroyOp = nullptr;
    for (mlir::Operation *useOp : elementResult.getUsers())
      if (mlir::isa<hlfir::DestroyOp>(useOp)) {
        if (destroyOp)
          fir::emitFatalError(loc,
                              "multiple DestroyOp's for ac-value expression");
        destroyOp = useOp;
      }

    if (destroyOp)
      destroyOp->erase();

    builder.create<hlfir::YieldElementOp>(loc, elementResult);
  }

  // Override the default, because the context scope must be popped in
  // pushValue().
  virtual void endImpliedDoScope() override { symMap.popImpliedDoBinding(); }

  /// Return the created hlfir.elemental.
  hlfir::Entity finishArrayCtorLowering(mlir::Location loc,
                                        fir::FirOpBuilder &builder) {
    return hlfir::Entity{elementalOp};
  }

private:
  mlir::Value shape;
  llvm::SmallVector<mlir::Value> lengthParams;
  hlfir::ExprType exprType;
  hlfir::ElementalOp elementalOp{};
};

/// Class that implements the "runtime temp strategy" to lower array
/// constructors.
class RuntimeTempStrategy : public StrategyBase {
  /// Name that will be given to the temporary allocation and hlfir.declare in
  /// the IR.
  static constexpr char tempName[] = ".tmp.arrayctor";

public:
  /// Start lowering an array constructor according to the runtime strategy.
  /// The temporary is only created if the extents and length parameters are
  /// already known. Otherwise, the handling of the allocation (and
  /// reallocation) is left up to the runtime.
  /// \p extent is the pre-computed extent of the array constructor, if it could
  /// be pre-computed. It is std::nullopt otherwise.
  /// \p lengths are the pre-computed length parameters of the array
  /// constructor, if they could be precomputed. \p missingLengthParameters is
  /// set to true if the length parameters could not be precomputed.
  RuntimeTempStrategy(mlir::Location loc, fir::FirOpBuilder &builder,
                      Fortran::lower::StatementContext &stmtCtx,
                      Fortran::lower::SymMap &symMap,
                      fir::SequenceType declaredType,
                      std::optional<mlir::Value> extent,
                      llvm::ArrayRef<mlir::Value> lengths,
                      bool missingLengthParameters)
      : StrategyBase{stmtCtx, symMap},
        arrayConstructorElementType{declaredType.getEleTy()} {
    mlir::Type heapType = fir::HeapType::get(declaredType);
    mlir::Type boxType = fir::BoxType::get(heapType);
    allocatableTemp = builder.createTemporary(loc, boxType, tempName);
    mlir::Value initialBoxValue;
    if (extent && !missingLengthParameters) {
      llvm::SmallVector<mlir::Value, 1> extents{*extent};
      mlir::Value tempStorage = builder.createHeapTemporary(
          loc, declaredType, tempName, extents, lengths);
      mlir::Value shape = builder.genShape(loc, extents);
      declare = builder.create<hlfir::DeclareOp>(
          loc, tempStorage, tempName, shape, lengths,
          /*dummy_scope=*/nullptr, fir::FortranVariableFlagsAttr{});
      initialBoxValue =
          builder.createBox(loc, boxType, declare->getOriginalBase(), shape,
                            /*slice=*/mlir::Value{}, lengths, /*tdesc=*/{});
    } else {
      // The runtime will have to do the initial allocation.
      // The declare operation cannot be emitted in this case since the final
      // array constructor has not yet been allocated. Instead, the resulting
      // temporary variable will be extracted from the allocatable descriptor
      // after all the API calls.
      // Prepare the initial state of the allocatable descriptor with a
      // deallocated status and all the available knowledge about the extent
      // and length parameters.
      llvm::SmallVector<mlir::Value> emboxLengths(lengths.begin(),
                                                  lengths.end());
      if (!extent)
        extent = builder.createIntegerConstant(loc, builder.getIndexType(), 0);
      if (missingLengthParameters) {
        if (mlir::isa<fir::CharacterType>(declaredType.getEleTy()))
          emboxLengths.push_back(builder.createIntegerConstant(
              loc, builder.getCharacterLengthType(), 0));
        else
          TODO(loc,
               "parametrized derived type array constructor without type-spec");
      }
      mlir::Value nullAddr = builder.createNullConstant(loc, heapType);
      mlir::Value shape = builder.genShape(loc, {*extent});
      initialBoxValue = builder.createBox(loc, boxType, nullAddr, shape,
                                          /*slice=*/mlir::Value{}, emboxLengths,
                                          /*tdesc=*/{});
    }
    builder.create<fir::StoreOp>(loc, initialBoxValue, allocatableTemp);
    arrayConstructorVector = fir::runtime::genInitArrayConstructorVector(
        loc, builder, allocatableTemp,
        builder.createBool(loc, missingLengthParameters));
  }

  bool useSimplePushRuntime(hlfir::Entity value) {
    return value.isScalar() &&
           !mlir::isa<fir::CharacterType>(arrayConstructorElementType) &&
           !fir::isRecordWithAllocatableMember(arrayConstructorElementType) &&
           !fir::isRecordWithTypeParameters(arrayConstructorElementType);
  }

  /// Push a lowered ac-value into the array constructor vector using
  /// the runtime API.
  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 hlfir::Entity value) {
    if (useSimplePushRuntime(value)) {
      auto [addrExv, cleanUp] = hlfir::convertToAddress(
          loc, builder, value, arrayConstructorElementType);
      mlir::Value addr = fir::getBase(addrExv);
      if (mlir::isa<fir::BaseBoxType>(addr.getType()))
        addr = builder.create<fir::BoxAddrOp>(loc, addr);
      fir::runtime::genPushArrayConstructorSimpleScalar(
          loc, builder, arrayConstructorVector, addr);
      if (cleanUp)
        (*cleanUp)();
      return;
    }
    auto [boxExv, cleanUp] =
        hlfir::convertToBox(loc, builder, value, arrayConstructorElementType);
    fir::runtime::genPushArrayConstructorValue(
        loc, builder, arrayConstructorVector, fir::getBase(boxExv));
    if (cleanUp)
      (*cleanUp)();
  }

  /// Start a fir.do_loop with the control from an implied-do and return
  /// the loop induction variable that is the ac-do-variable value.
  mlir::Value startImpliedDo(mlir::Location loc, fir::FirOpBuilder &builder,
                             mlir::Value lower, mlir::Value upper,
                             mlir::Value stride) {
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
    // Temp is created using createHeapTemporary, or allocated on the heap
    // by the runtime.
    mlir::Value mustFree = builder.createBool(loc, true);
    mlir::Value temp;
    if (declare)
      temp = declare->getBase();
    else
      temp = hlfir::derefPointersAndAllocatables(
          loc, builder, hlfir::Entity{allocatableTemp});
    auto hlfirExpr = builder.create<hlfir::AsExprOp>(loc, temp, mustFree);
    return hlfir::Entity{hlfirExpr};
  }

private:
  /// Element type of the array constructor being built.
  mlir::Type arrayConstructorElementType;
  /// Allocatable descriptor for the storage of the array constructor being
  /// built.
  mlir::Value allocatableTemp;
  /// Structure that allows the runtime API to maintain the status of
  /// of the array constructor being built between two API calls.
  mlir::Value arrayConstructorVector;
  /// DeclareOp for the array constructor storage, if it was possible to
  /// allocate it before any API calls.
  std::optional<hlfir::DeclareOp> declare;
};

/// Wrapper class that dispatch to the selected array constructor lowering
/// strategy and does nothing else.
class ArrayCtorLoweringStrategy {
public:
  template <typename A>
  ArrayCtorLoweringStrategy(A &&impl) : implVariant{std::forward<A>(impl)} {}

  void pushValue(mlir::Location loc, fir::FirOpBuilder &builder,
                 hlfir::Entity value) {
    return Fortran::common::visit(
        [&](auto &impl) { return impl.pushValue(loc, builder, value); },
        implVariant);
  }

  mlir::Value startImpliedDo(mlir::Location loc, fir::FirOpBuilder &builder,
                             mlir::Value lower, mlir::Value upper,
                             mlir::Value stride) {
    return Fortran::common::visit(
        [&](auto &impl) {
          return impl.startImpliedDo(loc, builder, lower, upper, stride);
        },
        implVariant);
  }

  hlfir::Entity finishArrayCtorLowering(mlir::Location loc,
                                        fir::FirOpBuilder &builder) {
    return Fortran::common::visit(
        [&](auto &impl) { return impl.finishArrayCtorLowering(loc, builder); },
        implVariant);
  }

  void startImpliedDoScope(llvm::StringRef doName, mlir::Value indexValue) {
    Fortran::common::visit(
        [&](auto &impl) {
          return impl.startImpliedDoScope(doName, indexValue);
        },
        implVariant);
  }

  void endImpliedDoScope() {
    Fortran::common::visit([&](auto &impl) { return impl.endImpliedDoScope(); },
                           implVariant);
  }

private:
  std::variant<InlinedTempStrategy, LooplessInlinedTempStrategy,
               AsElementalStrategy, RuntimeTempStrategy>
      implVariant;
};
} // namespace

//===----------------------------------------------------------------------===//
//   Definition of selectArrayCtorLoweringStrategy and its helpers.
//   This is the code that analyses the evaluate::ArrayConstructor<T>,
//   pre-lowers the array constructor extent and length parameters if it can,
//   and chooses the lowering strategy.
//===----------------------------------------------------------------------===//

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
    // Array constructors cannot be unlimited polymorphic (C7113), so there must
    // be a derived type spec available.
    return Fortran::lower::translateDerivedTypeToFIRType(
        converter, arrayCtorExpr.result().derivedTypeSpec());
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
    llvm::SmallVector<Fortran::lower::LenParameterTy> typeLengths;
    if (const Fortran::evaluate::ExtentExpr *lenExpr = arrayCtorExpr.LEN()) {
      lengths.push_back(
          lowerExtentExpr(loc, converter, symMap, stmtCtx, *lenExpr));
      if (std::optional<std::int64_t> cstLen =
              Fortran::evaluate::ToInt64(*lenExpr))
        typeLengths.push_back(*cstLen);
    }
    return Fortran::lower::getFIRType(&converter.getMLIRContext(),
                                      Fortran::common::TypeCategory::Character,
                                      Kind, typeLengths);
  }
};
} // namespace

/// Does the array constructor have length parameters that
/// LengthAndTypeCollector::collect could not lower because this requires
/// lowering an ac-value and must be delayed?
static bool missingLengthParameters(mlir::Type elementType,
                                    llvm::ArrayRef<mlir::Value> lengths) {
  return (mlir::isa<fir::CharacterType>(elementType) ||
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
      Fortran::evaluate::FoldingContext &,
      const Fortran::evaluate::ArrayConstructor<T> &arrayCtorExpr);

  // Can the array constructor easily be rewritten into an hlfir.elemental ?
  bool isSingleImpliedDoWithOneScalarPureExpr() const {
    return !anyArrayExpr && isPerfectLoopNest &&
           innerNumberOfExprIfPrefectNest == 1 && depthIfPerfectLoopNest == 1 &&
           innerExprIsPureIfPerfectNest;
  }

  bool anyImpliedDo = false;
  bool anyArrayExpr = false;
  bool isPerfectLoopNest = true;
  bool innerExprIsPureIfPerfectNest = false;
  std::int64_t innerNumberOfExprIfPrefectNest = 0;
  std::int64_t depthIfPerfectLoopNest = 0;
};
} // namespace

template <typename T>
ArrayCtorAnalysis::ArrayCtorAnalysis(
    Fortran::evaluate::FoldingContext &foldingContext,
    const Fortran::evaluate::ArrayConstructor<T> &arrayCtorExpr) {
  llvm::SmallVector<const Fortran::evaluate::ArrayConstructorValues<T> *>
      arrayValueListStack{&arrayCtorExpr};
  // Loop through the ac-value-list(s) of the array constructor.
  while (!arrayValueListStack.empty()) {
    std::int64_t localNumberOfImpliedDo = 0;
    std::int64_t localNumberOfExpr = 0;
    // Loop though the ac-value of an ac-value list, and add any nested
    // ac-value-list of ac-implied-do to the stack.
    const Fortran::evaluate::ArrayConstructorValues<T> *currentArrayValueList =
        arrayValueListStack.pop_back_val();
    for (const Fortran::evaluate::ArrayConstructorValue<T> &acValue :
         *currentArrayValueList)
      Fortran::common::visit(
          Fortran::common::visitors{
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
      if (isPerfectLoopNest) {
        // This this the only leaf of the array-constructor (the array
        // constructor is a nest of single implied-do with a list of expression
        // in the last deeper implied do). e.g: "[((i+j, i=1,n)j=1,m)]".
        innerNumberOfExprIfPrefectNest = localNumberOfExpr;
        if (localNumberOfExpr == 1)
          innerExprIsPureIfPerfectNest = !Fortran::evaluate::FindImpureCall(
              foldingContext, toEvExpr(std::get<Fortran::evaluate::Expr<T>>(
                                  currentArrayValueList->begin()->u)));
      }
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
  auto shapeExpr = Fortran::evaluate::GetContextFreeShape(
      converter.getFoldingContext(), arrayCtorExpr);
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
  ArrayCtorAnalysis analysis(converter.getFoldingContext(), arrayCtorExpr);
  bool needToEvaluateOneExprToGetLengthParameters =
      missingLengthParameters(elementType, lengths);
  auto declaredType = fir::SequenceType::get({typeExtent}, elementType);

  // Based on what was gathered and the result of the analysis, select and
  // instantiate the right lowering strategy for the array constructor.
  if (!extent || needToEvaluateOneExprToGetLengthParameters ||
      analysis.anyArrayExpr ||
      mlir::isa<fir::RecordType>(declaredType.getEleTy()))
    return RuntimeTempStrategy(
        loc, builder, stmtCtx, symMap, declaredType,
        extent ? std::optional<mlir::Value>(extent) : std::nullopt, lengths,
        needToEvaluateOneExprToGetLengthParameters);
  // Note: the generated hlfir.elemental is always unordered, thus,
  // AsElementalStrategy can only be used for array constructors without
  // impure ac-value expressions. If/when this changes, make sure
  // the 'unordered' attribute is set accordingly for the hlfir.elemental.
  if (analysis.isSingleImpliedDoWithOneScalarPureExpr())
    return AsElementalStrategy(loc, builder, stmtCtx, symMap, declaredType,
                               extent, lengths);

  if (analysis.anyImpliedDo)
    return InlinedTempStrategy(loc, builder, stmtCtx, symMap, declaredType,
                               extent, lengths);

  return LooplessInlinedTempStrategy(loc, builder, stmtCtx, symMap,
                                     declaredType, extent, lengths);
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
  arrayBuilder.startImpliedDoScope(toStringRef(impledDo.name()),
                                   impliedDoIndexValue);

  for (const auto &acValue : impledDo.values())
    Fortran::common::visit(
        [&](const auto &x) {
          genAcValue(loc, converter, x, symMap, stmtCtx, arrayBuilder);
        },
        acValue.u);

  arrayBuilder.endImpliedDoScope();
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
    Fortran::common::visit(
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
