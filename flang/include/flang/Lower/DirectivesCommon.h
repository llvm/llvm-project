//===-- Lower/DirectivesCommon.h --------------------------------*- C++ -*-===//
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
///
/// A location to place directive utilities shared across multiple lowering
/// files, e.g. utilities shared in OpenMP and OpenACC. The header file can
/// be used for both declarations and templated/inline implementations
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_DIRECTIVES_COMMON_H
#define FORTRAN_LOWER_DIRECTIVES_COMMON_H

#include "flang/Common/idioms.h"
#include "flang/Evaluate/tools.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/OpenACC.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include <list>
#include <type_traits>

namespace Fortran {
namespace lower {

/// Create empty blocks for the current region.
/// These blocks replace blocks parented to an enclosing region.
template <typename... TerminatorOps>
void createEmptyRegionBlocks(
    fir::FirOpBuilder &builder,
    std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
  mlir::Region *region = &builder.getRegion();
  for (Fortran::lower::pft::Evaluation &eval : evaluationList) {
    if (eval.block) {
      if (eval.block->empty()) {
        eval.block->erase();
        eval.block = builder.createBlock(region);
      } else {
        [[maybe_unused]] mlir::Operation &terminatorOp = eval.block->back();
        assert(mlir::isa<TerminatorOps...>(terminatorOp) &&
               "expected terminator op");
      }
    }
    if (!eval.isDirective() && eval.hasNestedEvaluations())
      createEmptyRegionBlocks<TerminatorOps...>(builder,
                                                eval.getNestedEvaluations());
  }
}

inline fir::factory::AddrAndBoundsInfo
getDataOperandBaseAddr(Fortran::lower::AbstractConverter &converter,
                       fir::FirOpBuilder &builder,
                       Fortran::lower::SymbolRef sym, mlir::Location loc,
                       bool unwrapFirBox = true) {
  return fir::factory::getDataOperandBaseAddr(
      builder, converter.getSymbolAddress(sym),
      Fortran::semantics::IsOptional(sym), loc, unwrapFirBox);
}

namespace detail {
template <typename T> //
static T &&AsRvalueRef(T &&t) {
  return std::move(t);
}
template <typename T> //
static T AsRvalueRef(T &t) {
  return t;
}
template <typename T> //
static T AsRvalueRef(const T &t) {
  return t;
}

// Helper class for stripping enclosing parentheses and a conversion that
// preserves type category. This is used for triplet elements, which are
// always of type integer(kind=8). The lower/upper bounds are converted to
// an "index" type, which is 64-bit, so the explicit conversion to kind=8
// (if present) is not needed. When it's present, though, it causes generated
// names to contain "int(..., kind=8)".
struct PeelConvert {
  template <Fortran::common::TypeCategory Category, int Kind>
  static Fortran::semantics::MaybeExpr visit_with_category(
      const Fortran::evaluate::Expr<Fortran::evaluate::Type<Category, Kind>>
          &expr) {
    return Fortran::common::visit(
        [](auto &&s) { return visit_with_category<Category, Kind>(s); },
        expr.u);
  }
  template <Fortran::common::TypeCategory Category, int Kind>
  static Fortran::semantics::MaybeExpr visit_with_category(
      const Fortran::evaluate::Convert<Fortran::evaluate::Type<Category, Kind>,
                                       Category> &expr) {
    return AsGenericExpr(AsRvalueRef(expr.left()));
  }
  template <Fortran::common::TypeCategory Category, int Kind, typename T>
  static Fortran::semantics::MaybeExpr visit_with_category(const T &) {
    return std::nullopt; //
  }
  template <Fortran::common::TypeCategory Category, typename T>
  static Fortran::semantics::MaybeExpr visit_with_category(const T &) {
    return std::nullopt; //
  }

  template <Fortran::common::TypeCategory Category>
  static Fortran::semantics::MaybeExpr
  visit(const Fortran::evaluate::Expr<Fortran::evaluate::SomeKind<Category>>
            &expr) {
    return Fortran::common::visit(
        [](auto &&s) { return visit_with_category<Category>(s); }, expr.u);
  }
  static Fortran::semantics::MaybeExpr
  visit(const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> &expr) {
    return Fortran::common::visit([](auto &&s) { return visit(s); }, expr.u);
  }
  template <typename T> //
  static Fortran::semantics::MaybeExpr visit(const T &) {
    return std::nullopt;
  }
};

static inline Fortran::semantics::SomeExpr
peelOuterConvert(Fortran::semantics::SomeExpr &expr) {
  if (auto peeled = PeelConvert::visit(expr))
    return *peeled;
  return expr;
}
} // namespace detail

/// Generate bounds operations for an array section when subscripts are
/// provided.
template <typename BoundsOp, typename BoundsType>
llvm::SmallVector<mlir::Value>
genBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
             Fortran::lower::AbstractConverter &converter,
             Fortran::lower::StatementContext &stmtCtx,
             const std::vector<Fortran::evaluate::Subscript> &subscripts,
             std::stringstream &asFortran, fir::ExtendedValue &dataExv,
             bool dataExvIsAssumedSize, fir::factory::AddrAndBoundsInfo &info,
             bool treatIndexAsSection = false,
             bool strideIncludeLowerExtent = false) {
  int dimension = 0;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Type boundTy = builder.getType<BoundsType>();
  llvm::SmallVector<mlir::Value> bounds;

  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  const int dataExvRank = static_cast<int>(dataExv.rank());
  mlir::Value cumulativeExtent = one;
  for (const auto &subscript : subscripts) {
    const auto *triplet{std::get_if<Fortran::evaluate::Triplet>(&subscript.u)};
    if (triplet || treatIndexAsSection) {
      if (dimension != 0)
        asFortran << ',';
      mlir::Value lbound, ubound, extent;
      std::optional<std::int64_t> lval, uval;
      mlir::Value baseLb =
          fir::factory::readLowerBound(builder, loc, dataExv, dimension, one);
      bool defaultLb = baseLb == one;
      mlir::Value stride = one;
      bool strideInBytes = false;

      if (mlir::isa<fir::BaseBoxType>(
              fir::unwrapRefType(info.addr.getType()))) {
        if (info.isPresent) {
          stride =
              builder
                  .genIfOp(loc, idxTy, info.isPresent, /*withElseRegion=*/true)
                  .genThen([&]() {
                    mlir::Value box =
                        !fir::isBoxAddress(info.addr.getType())
                            ? info.addr
                            : fir::LoadOp::create(builder, loc, info.addr);
                    mlir::Value d =
                        builder.createIntegerConstant(loc, idxTy, dimension);
                    auto dimInfo = fir::BoxDimsOp::create(builder, loc, idxTy,
                                                          idxTy, idxTy, box, d);
                    fir::ResultOp::create(builder, loc,
                                          dimInfo.getByteStride());
                  })
                  .genElse([&] {
                    mlir::Value zero =
                        builder.createIntegerConstant(loc, idxTy, 0);
                    fir::ResultOp::create(builder, loc, zero);
                  })
                  .getResults()[0];
        } else {
          mlir::Value box = !fir::isBoxAddress(info.addr.getType())
                                ? info.addr
                                : fir::LoadOp::create(builder, loc, info.addr);
          mlir::Value d = builder.createIntegerConstant(loc, idxTy, dimension);
          auto dimInfo =
              fir::BoxDimsOp::create(builder, loc, idxTy, idxTy, idxTy, box, d);
          stride = dimInfo.getByteStride();
        }
        strideInBytes = true;
      }

      Fortran::semantics::MaybeExpr lower;
      if (triplet) {
        lower = Fortran::evaluate::AsGenericExpr(triplet->lower());
      } else {
        // Case of IndirectSubscriptIntegerExpr
        using IndirectSubscriptIntegerExpr =
            Fortran::evaluate::IndirectSubscriptIntegerExpr;
        using SubscriptInteger = Fortran::evaluate::SubscriptInteger;
        Fortran::evaluate::Expr<SubscriptInteger> oneInt =
            std::get<IndirectSubscriptIntegerExpr>(subscript.u).value();
        lower = Fortran::evaluate::AsGenericExpr(std::move(oneInt));
        if (lower->Rank() > 0) {
          mlir::emitError(
              loc, "vector subscript cannot be used for an array section");
          break;
        }
      }
      if (lower) {
        lval = Fortran::evaluate::ToInt64(*lower);
        if (lval) {
          if (defaultLb) {
            lbound = builder.createIntegerConstant(loc, idxTy, *lval - 1);
          } else {
            mlir::Value lb = builder.createIntegerConstant(loc, idxTy, *lval);
            lbound = mlir::arith::SubIOp::create(builder, loc, lb, baseLb);
          }
          asFortran << *lval;
        } else {
          mlir::Value lb =
              fir::getBase(converter.genExprValue(loc, *lower, stmtCtx));
          lb = builder.createConvert(loc, baseLb.getType(), lb);
          lbound = mlir::arith::SubIOp::create(builder, loc, lb, baseLb);
          asFortran << detail::peelOuterConvert(*lower).AsFortran();
        }
      } else {
        // If the lower bound is not specified, then the section
        // starts from offset 0 of the dimension.
        // Note that the lowerbound in the BoundsOp is always 0-based.
        lbound = zero;
      }

      if (!triplet) {
        // If it is a scalar subscript, then the upper bound
        // is equal to the lower bound, and the extent is one.
        ubound = lbound;
        extent = one;
      } else {
        asFortran << ':';
        Fortran::semantics::MaybeExpr upper =
            Fortran::evaluate::AsGenericExpr(triplet->upper());

        if (upper) {
          uval = Fortran::evaluate::ToInt64(*upper);
          if (uval) {
            if (defaultLb) {
              ubound = builder.createIntegerConstant(loc, idxTy, *uval - 1);
            } else {
              mlir::Value ub = builder.createIntegerConstant(loc, idxTy, *uval);
              ubound = mlir::arith::SubIOp::create(builder, loc, ub, baseLb);
            }
            asFortran << *uval;
          } else {
            mlir::Value ub =
                fir::getBase(converter.genExprValue(loc, *upper, stmtCtx));
            ub = builder.createConvert(loc, baseLb.getType(), ub);
            ubound = mlir::arith::SubIOp::create(builder, loc, ub, baseLb);
            asFortran << detail::peelOuterConvert(*upper).AsFortran();
          }
        }
        if (lower && upper) {
          if (lval && uval && *uval < *lval) {
            mlir::emitError(loc, "zero sized array section");
            break;
          } else {
            // Stride is mandatory in evaluate::Triplet. Make sure it's 1.
            auto val = Fortran::evaluate::ToInt64(triplet->GetStride());
            if (!val || *val != 1) {
              mlir::emitError(loc, "stride cannot be specified on "
                                   "an array section");
              break;
            }
          }
        }

        if (info.isPresent && mlir::isa<fir::BaseBoxType>(
                                  fir::unwrapRefType(info.addr.getType()))) {
          extent =
              builder
                  .genIfOp(loc, idxTy, info.isPresent, /*withElseRegion=*/true)
                  .genThen([&]() {
                    mlir::Value ext = fir::factory::readExtent(
                        builder, loc, dataExv, dimension);
                    fir::ResultOp::create(builder, loc, ext);
                  })
                  .genElse([&] {
                    mlir::Value zero =
                        builder.createIntegerConstant(loc, idxTy, 0);
                    fir::ResultOp::create(builder, loc, zero);
                  })
                  .getResults()[0];
        } else {
          extent = fir::factory::readExtent(builder, loc, dataExv, dimension);
        }

        if (dataExvIsAssumedSize && dimension + 1 == dataExvRank) {
          extent = zero;
          if (ubound && lbound) {
            mlir::Value diff =
                mlir::arith::SubIOp::create(builder, loc, ubound, lbound);
            extent = mlir::arith::AddIOp::create(builder, loc, diff, one);
          }
          if (!ubound)
            ubound = lbound;
        }

        if (!ubound) {
          // ub = extent - 1
          ubound = mlir::arith::SubIOp::create(builder, loc, extent, one);
        }
      }

      // When the strideInBytes is true, it means the stride is from descriptor
      // and this already includes the lower extents.
      if (strideIncludeLowerExtent && !strideInBytes) {
        stride = cumulativeExtent;
        cumulativeExtent = builder.createOrFold<mlir::arith::MulIOp>(
            loc, cumulativeExtent, extent);
      }

      mlir::Value bound =
          BoundsOp::create(builder, loc, boundTy, lbound, ubound, extent,
                           stride, strideInBytes, baseLb);
      bounds.push_back(bound);
      ++dimension;
    }
  }
  return bounds;
}

namespace detail {
template <typename Ref, typename Expr> //
std::optional<Ref> getRef(Expr &&expr) {
  if constexpr (std::is_same_v<llvm::remove_cvref_t<Expr>,
                               Fortran::evaluate::DataRef>) {
    if (auto *ref = std::get_if<Ref>(&expr.u))
      return *ref;
    return std::nullopt;
  } else {
    auto maybeRef = Fortran::evaluate::ExtractDataRef(expr);
    if (!maybeRef || !std::holds_alternative<Ref>(maybeRef->u))
      return std::nullopt;
    return std::get<Ref>(maybeRef->u);
  }
}
} // namespace detail

template <typename BoundsOp, typename BoundsType>
fir::factory::AddrAndBoundsInfo gatherDataOperandAddrAndBounds(
    Fortran::lower::AbstractConverter &converter, fir::FirOpBuilder &builder,
    semantics::SemanticsContext &semaCtx,
    Fortran::lower::StatementContext &stmtCtx,
    Fortran::semantics::SymbolRef symbol,
    const Fortran::semantics::MaybeExpr &maybeDesignator,
    mlir::Location operandLocation, std::stringstream &asFortran,
    llvm::SmallVector<mlir::Value> &bounds, bool treatIndexAsSection = false,
    bool unwrapFirBox = true, bool genDefaultBounds = true,
    bool strideIncludeLowerExtent = false) {
  using namespace Fortran;

  fir::factory::AddrAndBoundsInfo info;

  if (!maybeDesignator) {
    info = getDataOperandBaseAddr(converter, builder, symbol, operandLocation,
                                  unwrapFirBox);
    asFortran << symbol->name().ToString();
    return info;
  }

  semantics::SomeExpr designator = *maybeDesignator;

  if ((designator.Rank() > 0 || treatIndexAsSection) &&
      IsArrayElement(designator)) {
    auto arrayRef = detail::getRef<evaluate::ArrayRef>(designator);
    // This shouldn't fail after IsArrayElement(designator).
    assert(arrayRef && "Expecting ArrayRef");

    fir::ExtendedValue dataExv;
    bool dataExvIsAssumedSize = false;

    auto toMaybeExpr = [&](auto &&base) {
      using BaseType = llvm::remove_cvref_t<decltype(base)>;
      evaluate::ExpressionAnalyzer ea{semaCtx};

      if constexpr (std::is_same_v<evaluate::NamedEntity, BaseType>) {
        if (auto *ref = base.UnwrapSymbolRef())
          return ea.Designate(evaluate::DataRef{*ref});
        if (auto *ref = base.UnwrapComponent())
          return ea.Designate(evaluate::DataRef{*ref});
        llvm_unreachable("Unexpected NamedEntity");
      } else {
        static_assert(std::is_same_v<semantics::SymbolRef, BaseType>);
        return ea.Designate(evaluate::DataRef{base});
      }
    };

    auto arrayBase = toMaybeExpr(arrayRef->base());
    assert(arrayBase);

    if (detail::getRef<evaluate::Component>(*arrayBase)) {
      dataExv = converter.genExprAddr(operandLocation, *arrayBase, stmtCtx);
      info.addr = fir::getBase(dataExv);
      info.rawInput = info.addr;
      asFortran << arrayBase->AsFortran();
    } else {
      const semantics::Symbol &sym = arrayRef->GetLastSymbol();
      dataExvIsAssumedSize =
          Fortran::semantics::IsAssumedSizeArray(sym.GetUltimate());
      info = getDataOperandBaseAddr(converter, builder, sym, operandLocation,
                                    unwrapFirBox);
      dataExv = converter.getSymbolExtendedValue(sym);
      asFortran << sym.name().ToString();
    }

    if (!arrayRef->subscript().empty()) {
      asFortran << '(';
      bounds = genBoundsOps<BoundsOp, BoundsType>(
          builder, operandLocation, converter, stmtCtx, arrayRef->subscript(),
          asFortran, dataExv, dataExvIsAssumedSize, info, treatIndexAsSection,
          strideIncludeLowerExtent);
    }
    asFortran << ')';
  } else if (auto compRef = detail::getRef<evaluate::Component>(designator)) {
    fir::ExtendedValue compExv =
        converter.genExprAddr(operandLocation, designator, stmtCtx);
    info.addr = fir::getBase(compExv);
    info.rawInput = info.addr;
    if (genDefaultBounds &&
        mlir::isa<fir::SequenceType>(fir::unwrapRefType(info.addr.getType())))
      bounds = fir::factory::genBaseBoundsOps<BoundsOp, BoundsType>(
          builder, operandLocation, compExv,
          /*isAssumedSize=*/false, strideIncludeLowerExtent);
    asFortran << designator.AsFortran();

    if (semantics::IsOptional(compRef->GetLastSymbol())) {
      info.isPresent = fir::IsPresentOp::create(
          builder, operandLocation, builder.getI1Type(), info.rawInput);
    }

    if (unwrapFirBox) {
      if (auto loadOp =
              mlir::dyn_cast_or_null<fir::LoadOp>(info.addr.getDefiningOp())) {
        if (fir::isAllocatableType(loadOp.getType()) ||
            fir::isPointerType(loadOp.getType())) {
          info.boxType = info.addr.getType();
          info.addr =
              fir::BoxAddrOp::create(builder, operandLocation, info.addr);
        }
        info.rawInput = info.addr;
      }
    }

    // If the component is an allocatable or pointer the result of
    // genExprAddr will be the result of a fir.box_addr operation or
    // a fir.box_addr has been inserted just before.
    // Retrieve the box so we handle it like other descriptor.
    if (auto boxAddrOp =
            mlir::dyn_cast_or_null<fir::BoxAddrOp>(info.addr.getDefiningOp())) {
      info.addr = boxAddrOp.getVal();
      info.boxType = info.addr.getType();
      info.rawInput = info.addr;
      if (genDefaultBounds)
        bounds = fir::factory::genBoundsOpsFromBox<BoundsOp, BoundsType>(
            builder, operandLocation, compExv, info);
    }
  } else {
    if (detail::getRef<evaluate::ArrayRef>(designator)) {
      fir::ExtendedValue compExv =
          converter.genExprAddr(operandLocation, designator, stmtCtx);
      info.addr = fir::getBase(compExv);
      info.rawInput = info.addr;
      asFortran << designator.AsFortran();
    } else if (auto symRef = detail::getRef<semantics::SymbolRef>(designator)) {
      // Scalar or full array.
      fir::ExtendedValue dataExv = converter.getSymbolExtendedValue(*symRef);
      info = getDataOperandBaseAddr(converter, builder, *symRef,
                                    operandLocation, unwrapFirBox);
      if (genDefaultBounds && mlir::isa<fir::BaseBoxType>(
                                  fir::unwrapRefType(info.addr.getType()))) {
        info.boxType = fir::unwrapRefType(info.addr.getType());
        bounds = fir::factory::genBoundsOpsFromBox<BoundsOp, BoundsType>(
            builder, operandLocation, dataExv, info);
      }
      bool dataExvIsAssumedSize =
          Fortran::semantics::IsAssumedSizeArray(symRef->get().GetUltimate());
      if (genDefaultBounds && mlir::isa<fir::SequenceType>(
                                  fir::unwrapRefType(info.addr.getType()))) {
        bounds = fir::factory::genBaseBoundsOps<BoundsOp, BoundsType>(
            builder, operandLocation, dataExv, dataExvIsAssumedSize,
            strideIncludeLowerExtent);
      }
      if (genDefaultBounds && fir::characterWithDynamicLen(
                                  fir::unwrapRefType(info.addr.getType())) ||
          mlir::isa<fir::BoxCharType>(
              fir::unwrapRefType(info.addr.getType()))) {
        bounds = {fir::factory::genBoundsOpFromBoxChar<BoundsOp, BoundsType>(
            builder, operandLocation, dataExv, info)};
      }
      asFortran << symRef->get().name().ToString();
    } else { // Unsupported
      llvm::report_fatal_error("Unsupported type of OpenACC operand");
    }
  }

  return info;
}

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_DIRECTIVES_COMMON_H
