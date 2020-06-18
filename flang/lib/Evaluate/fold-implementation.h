//===-- lib/Evaluate/fold-implementation.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_FOLD_IMPLEMENTATION_H_
#define FORTRAN_EVALUATE_FOLD_IMPLEMENTATION_H_

#include "character.h"
#include "host.h"
#include "int-power.h"
#include "intrinsics-library-templates.h"
#include "flang/Common/indirection.h"
#include "flang/Common/template.h"
#include "flang/Common/unwrap.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/constant.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/formatting.h"
#include "flang/Evaluate/shape.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/message.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <optional>
#include <type_traits>
#include <variant>

// Some environments, viz. clang on Darwin, allow the macro HUGE
// to leak out of <math.h> even when it is never directly included.
#undef HUGE

namespace Fortran::evaluate {

// Utilities
template <typename T> class Folder {
public:
  explicit Folder(FoldingContext &c) : context_{c} {}
  std::optional<Expr<T>> GetNamedConstantValue(const Symbol &);
  std::optional<Constant<T>> GetFoldedNamedConstantValue(const Symbol &);
  std::optional<Constant<T>> ApplySubscripts(const Constant<T> &array,
      const std::vector<Constant<SubscriptInteger>> &subscripts);
  std::optional<Constant<T>> ApplyComponent(Constant<SomeDerived> &&,
      const Symbol &component,
      const std::vector<Constant<SubscriptInteger>> * = nullptr);
  std::optional<Constant<T>> GetConstantComponent(
      Component &, const std::vector<Constant<SubscriptInteger>> * = nullptr);
  std::optional<Constant<T>> Folding(ArrayRef &);
  Expr<T> Folding(Designator<T> &&);
  Constant<T> *Folding(std::optional<ActualArgument> &);
  Expr<T> Reshape(FunctionRef<T> &&);

private:
  FoldingContext &context_;
};

std::optional<Constant<SubscriptInteger>> GetConstantSubscript(
    FoldingContext &, Subscript &, const NamedEntity &, int dim);

// FoldOperation() rewrites expression tree nodes.
// If there is any possibility that the rewritten node will
// not have the same representation type, the result of
// FoldOperation() will be packaged in an Expr<> of the same
// specific type.

// no-op base case
template <typename A>
common::IfNoLvalue<Expr<ResultType<A>>, A> FoldOperation(
    FoldingContext &, A &&x) {
  static_assert(!std::is_same_v<A, Expr<ResultType<A>>>,
      "call Fold() instead for Expr<>");
  return Expr<ResultType<A>>{std::move(x)};
}

Component FoldOperation(FoldingContext &, Component &&);
NamedEntity FoldOperation(FoldingContext &, NamedEntity &&);
Triplet FoldOperation(FoldingContext &, Triplet &&);
Subscript FoldOperation(FoldingContext &, Subscript &&);
ArrayRef FoldOperation(FoldingContext &, ArrayRef &&);
CoarrayRef FoldOperation(FoldingContext &, CoarrayRef &&);
DataRef FoldOperation(FoldingContext &, DataRef &&);
Substring FoldOperation(FoldingContext &, Substring &&);
ComplexPart FoldOperation(FoldingContext &, ComplexPart &&);

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, FunctionRef<T> &&);
template <int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Integer, KIND>> &&);
template <int KIND>
Expr<Type<TypeCategory::Real, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Real, KIND>> &&);
template <int KIND>
Expr<Type<TypeCategory::Complex, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Complex, KIND>> &&);
template <int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldIntrinsicFunction(
    FoldingContext &context, FunctionRef<Type<TypeCategory::Logical, KIND>> &&);

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, Designator<T> &&designator) {
  return Folder<T>{context}.Folding(std::move(designator));
}

template <int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldOperation(
    FoldingContext &, TypeParamInquiry<KIND> &&);
Expr<ImpliedDoIndex::Result> FoldOperation(
    FoldingContext &context, ImpliedDoIndex &&);
template <typename T>
Expr<T> FoldOperation(FoldingContext &, ArrayConstructor<T> &&);
Expr<SomeDerived> FoldOperation(FoldingContext &, StructureConstructor &&);

template <typename T>
std::optional<Expr<T>> Folder<T>::GetNamedConstantValue(const Symbol &symbol0) {
  const Symbol &symbol{ResolveAssociations(symbol0)};
  if (IsNamedConstant(symbol)) {
    if (const auto *object{
            symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
      if (object->initWasValidated()) {
        const auto *constant{UnwrapConstantValue<T>(object->init())};
        return Expr<T>{DEREF(constant)};
      }
      if (const auto &init{object->init()}) {
        if (auto dyType{DynamicType::From(symbol)}) {
          semantics::ObjectEntityDetails *mutableObject{
              const_cast<semantics::ObjectEntityDetails *>(object)};
          auto converted{
              ConvertToType(*dyType, std::move(mutableObject->init().value()))};
          // Reset expression now to prevent infinite loops if the init
          // expression depends on symbol itself.
          mutableObject->set_init(std::nullopt);
          if (converted) {
            *converted = Fold(context_, std::move(*converted));
            auto *unwrapped{UnwrapExpr<Expr<T>>(*converted)};
            CHECK(unwrapped);
            if (auto *constant{UnwrapConstantValue<T>(*unwrapped)}) {
              if (symbol.Rank() > 0) {
                if (constant->Rank() == 0) {
                  // scalar expansion
                  if (auto symShape{GetShape(context_, symbol)}) {
                    if (auto extents{AsConstantExtents(context_, *symShape)}) {
                      *constant = constant->Reshape(std::move(*extents));
                      CHECK(constant->Rank() == symbol.Rank());
                    }
                  }
                }
                if (constant->Rank() == symbol.Rank()) {
                  NamedEntity base{symbol};
                  if (auto lbounds{AsConstantExtents(
                          context_, GetLowerBounds(context_, base))}) {
                    constant->set_lbounds(*std::move(lbounds));
                  }
                }
              }
              mutableObject->set_init(AsGenericExpr(Expr<T>{*constant}));
              if (auto constShape{GetShape(context_, *constant)}) {
                if (auto symShape{GetShape(context_, symbol)}) {
                  if (CheckConformance(context_.messages(), *constShape,
                          *symShape, "initialization expression",
                          "PARAMETER")) {
                    mutableObject->set_initWasValidated();
                    return std::move(*unwrapped);
                  }
                } else {
                  context_.messages().Say(symbol.name(),
                      "Could not determine the shape of the PARAMETER"_err_en_US);
                }
              } else {
                context_.messages().Say(symbol.name(),
                    "Could not determine the shape of the initialization expression"_err_en_US);
              }
              mutableObject->set_init(std::nullopt);
            } else {
              context_.messages().Say(symbol.name(),
                  "Initialization expression for PARAMETER '%s' (%s) cannot be computed as a constant value"_err_en_US,
                  symbol.name(), unwrapped->AsFortran());
            }
          } else {
            context_.messages().Say(symbol.name(),
                "Initialization expression for PARAMETER '%s' (%s) cannot be converted to its type (%s)"_err_en_US,
                symbol.name(), init->AsFortran(), dyType->AsFortran());
          }
        }
      }
    }
  }
  return std::nullopt;
}

template <typename T>
std::optional<Constant<T>> Folder<T>::GetFoldedNamedConstantValue(
    const Symbol &symbol) {
  if (auto value{GetNamedConstantValue(symbol)}) {
    Expr<T> folded{Fold(context_, std::move(*value))};
    if (const Constant<T> *value{UnwrapConstantValue<T>(folded)}) {
      return *value;
    }
  }
  return std::nullopt;
}

template <typename T>
std::optional<Constant<T>> Folder<T>::Folding(ArrayRef &aRef) {
  std::vector<Constant<SubscriptInteger>> subscripts;
  int dim{0};
  for (Subscript &ss : aRef.subscript()) {
    if (auto constant{GetConstantSubscript(context_, ss, aRef.base(), dim++)}) {
      subscripts.emplace_back(std::move(*constant));
    } else {
      return std::nullopt;
    }
  }
  if (Component * component{aRef.base().UnwrapComponent()}) {
    return GetConstantComponent(*component, &subscripts);
  } else if (std::optional<Constant<T>> array{
                 GetFoldedNamedConstantValue(aRef.base().GetLastSymbol())}) {
    return ApplySubscripts(*array, subscripts);
  } else {
    return std::nullopt;
  }
}

template <typename T>
std::optional<Constant<T>> Folder<T>::ApplySubscripts(const Constant<T> &array,
    const std::vector<Constant<SubscriptInteger>> &subscripts) {
  const auto &shape{array.shape()};
  const auto &lbounds{array.lbounds()};
  int rank{GetRank(shape)};
  CHECK(rank == static_cast<int>(subscripts.size()));
  std::size_t elements{1};
  ConstantSubscripts resultShape;
  ConstantSubscripts ssLB;
  for (const auto &ss : subscripts) {
    CHECK(ss.Rank() <= 1);
    if (ss.Rank() == 1) {
      resultShape.push_back(static_cast<ConstantSubscript>(ss.size()));
      elements *= ss.size();
      ssLB.push_back(ss.lbounds().front());
    }
  }
  ConstantSubscripts ssAt(rank, 0), at(rank, 0), tmp(1, 0);
  std::vector<Scalar<T>> values;
  while (elements-- > 0) {
    bool increment{true};
    int k{0};
    for (int j{0}; j < rank; ++j) {
      if (subscripts[j].Rank() == 0) {
        at[j] = subscripts[j].GetScalarValue().value().ToInt64();
      } else {
        CHECK(k < GetRank(resultShape));
        tmp[0] = ssLB.at(k) + ssAt.at(k);
        at[j] = subscripts[j].At(tmp).ToInt64();
        if (increment) {
          if (++ssAt[k] == resultShape[k]) {
            ssAt[k] = 0;
          } else {
            increment = false;
          }
        }
        ++k;
      }
      if (at[j] < lbounds[j] || at[j] >= lbounds[j] + shape[j]) {
        context_.messages().Say(
            "Subscript value (%jd) is out of range on dimension %d in reference to a constant array value"_err_en_US,
            at[j], j + 1);
        return std::nullopt;
      }
    }
    values.emplace_back(array.At(at));
    CHECK(!increment || elements == 0);
    CHECK(k == GetRank(resultShape));
  }
  if constexpr (T::category == TypeCategory::Character) {
    return Constant<T>{array.LEN(), std::move(values), std::move(resultShape)};
  } else if constexpr (std::is_same_v<T, SomeDerived>) {
    return Constant<T>{array.result().derivedTypeSpec(), std::move(values),
        std::move(resultShape)};
  } else {
    return Constant<T>{std::move(values), std::move(resultShape)};
  }
}

template <typename T>
std::optional<Constant<T>> Folder<T>::ApplyComponent(
    Constant<SomeDerived> &&structures, const Symbol &component,
    const std::vector<Constant<SubscriptInteger>> *subscripts) {
  if (auto scalar{structures.GetScalarValue()}) {
    if (auto *expr{scalar->Find(component)}) {
      if (const Constant<T> *value{UnwrapConstantValue<T>(*expr)}) {
        if (!subscripts) {
          return std::move(*value);
        } else {
          return ApplySubscripts(*value, *subscripts);
        }
      }
    }
  } else {
    // A(:)%scalar_component & A(:)%array_component(subscripts)
    std::unique_ptr<ArrayConstructor<T>> array;
    if (structures.empty()) {
      return std::nullopt;
    }
    ConstantSubscripts at{structures.lbounds()};
    do {
      StructureConstructor scalar{structures.At(at)};
      if (auto *expr{scalar.Find(component)}) {
        if (const Constant<T> *value{UnwrapConstantValue<T>(*expr)}) {
          if (!array.get()) {
            // This technique ensures that character length or derived type
            // information is propagated to the array constructor.
            auto *typedExpr{UnwrapExpr<Expr<T>>(*expr)};
            CHECK(typedExpr);
            array = std::make_unique<ArrayConstructor<T>>(*typedExpr);
          }
          if (subscripts) {
            if (auto element{ApplySubscripts(*value, *subscripts)}) {
              CHECK(element->Rank() == 0);
              array->Push(Expr<T>{std::move(*element)});
            } else {
              return std::nullopt;
            }
          } else {
            CHECK(value->Rank() == 0);
            array->Push(Expr<T>{*value});
          }
        } else {
          return std::nullopt;
        }
      }
    } while (structures.IncrementSubscripts(at));
    // Fold the ArrayConstructor<> into a Constant<>.
    CHECK(array);
    Expr<T> result{Fold(context_, Expr<T>{std::move(*array)})};
    if (auto *constant{UnwrapConstantValue<T>(result)}) {
      return constant->Reshape(common::Clone(structures.shape()));
    }
  }
  return std::nullopt;
}

template <typename T>
std::optional<Constant<T>> Folder<T>::GetConstantComponent(Component &component,
    const std::vector<Constant<SubscriptInteger>> *subscripts) {
  if (std::optional<Constant<SomeDerived>> structures{std::visit(
          common::visitors{
              [&](const Symbol &symbol) {
                return Folder<SomeDerived>{context_}
                    .GetFoldedNamedConstantValue(symbol);
              },
              [&](ArrayRef &aRef) {
                return Folder<SomeDerived>{context_}.Folding(aRef);
              },
              [&](Component &base) {
                return Folder<SomeDerived>{context_}.GetConstantComponent(base);
              },
              [&](CoarrayRef &) {
                return std::optional<Constant<SomeDerived>>{};
              },
          },
          component.base().u)}) {
    return ApplyComponent(
        std::move(*structures), component.GetLastSymbol(), subscripts);
  } else {
    return std::nullopt;
  }
}

template <typename T> Expr<T> Folder<T>::Folding(Designator<T> &&designator) {
  if constexpr (T::category == TypeCategory::Character) {
    if (auto *substring{common::Unwrap<Substring>(designator.u)}) {
      if (std::optional<Expr<SomeCharacter>> folded{
              substring->Fold(context_)}) {
        if (auto value{GetScalarConstantValue<T>(*folded)}) {
          return Expr<T>{*value};
        }
      }
      if (auto length{ToInt64(Fold(context_, substring->LEN()))}) {
        if (*length == 0) {
          return Expr<T>{Constant<T>{Scalar<T>{}}};
        }
      }
    }
  }
  return std::visit(
      common::visitors{
          [&](SymbolRef &&symbol) {
            if (auto constant{GetFoldedNamedConstantValue(*symbol)}) {
              return Expr<T>{std::move(*constant)};
            }
            return Expr<T>{std::move(designator)};
          },
          [&](ArrayRef &&aRef) {
            aRef = FoldOperation(context_, std::move(aRef));
            if (auto c{Folding(aRef)}) {
              return Expr<T>{std::move(*c)};
            } else {
              return Expr<T>{Designator<T>{std::move(aRef)}};
            }
          },
          [&](Component &&component) {
            component = FoldOperation(context_, std::move(component));
            if (auto c{GetConstantComponent(component)}) {
              return Expr<T>{std::move(*c)};
            } else {
              return Expr<T>{Designator<T>{std::move(component)}};
            }
          },
          [&](auto &&x) {
            return Expr<T>{
                Designator<T>{FoldOperation(context_, std::move(x))}};
          },
      },
      std::move(designator.u));
}

// Apply type conversion and re-folding if necessary.
// This is where BOZ arguments are converted.
template <typename T>
Constant<T> *Folder<T>::Folding(std::optional<ActualArgument> &arg) {
  if (auto *expr{UnwrapExpr<Expr<SomeType>>(arg)}) {
    if (!UnwrapExpr<Expr<T>>(*expr)) {
      if (auto converted{ConvertToType(T::GetType(), std::move(*expr))}) {
        *expr = Fold(context_, std::move(*converted));
      }
    }
    return UnwrapConstantValue<T>(*expr);
  }
  return nullptr;
}

template <typename... A, std::size_t... I>
std::optional<std::tuple<const Constant<A> *...>> GetConstantArgumentsHelper(
    FoldingContext &context, ActualArguments &arguments,
    std::index_sequence<I...>) {
  static_assert(
      (... && IsSpecificIntrinsicType<A>)); // TODO derived types for MERGE?
  static_assert(sizeof...(A) > 0);
  std::tuple<const Constant<A> *...> args{
      Folder<A>{context}.Folding(arguments.at(I))...};
  if ((... && (std::get<I>(args)))) {
    return args;
  } else {
    return std::nullopt;
  }
}

template <typename... A>
std::optional<std::tuple<const Constant<A> *...>> GetConstantArguments(
    FoldingContext &context, ActualArguments &args) {
  return GetConstantArgumentsHelper<A...>(
      context, args, std::index_sequence_for<A...>{});
}

template <typename... A, std::size_t... I>
std::optional<std::tuple<Scalar<A>...>> GetScalarConstantArgumentsHelper(
    FoldingContext &context, ActualArguments &args, std::index_sequence<I...>) {
  if (auto constArgs{GetConstantArguments<A...>(context, args)}) {
    return std::tuple<Scalar<A>...>{
        std::get<I>(*constArgs)->GetScalarValue().value()...};
  } else {
    return std::nullopt;
  }
}

template <typename... A>
std::optional<std::tuple<Scalar<A>...>> GetScalarConstantArguments(
    FoldingContext &context, ActualArguments &args) {
  return GetScalarConstantArgumentsHelper<A...>(
      context, args, std::index_sequence_for<A...>{});
}

// helpers to fold intrinsic function references
// Define callable types used in a common utility that
// takes care of array and cast/conversion aspects for elemental intrinsics

template <typename TR, typename... TArgs>
using ScalarFunc = std::function<Scalar<TR>(const Scalar<TArgs> &...)>;
template <typename TR, typename... TArgs>
using ScalarFuncWithContext =
    std::function<Scalar<TR>(FoldingContext &, const Scalar<TArgs> &...)>;

template <template <typename, typename...> typename WrapperType, typename TR,
    typename... TA, std::size_t... I>
Expr<TR> FoldElementalIntrinsicHelper(FoldingContext &context,
    FunctionRef<TR> &&funcRef, WrapperType<TR, TA...> func,
    std::index_sequence<I...>) {
  if (std::optional<std::tuple<const Constant<TA> *...>> args{
          GetConstantArguments<TA...>(context, funcRef.arguments())}) {
    // Compute the shape of the result based on shapes of arguments
    ConstantSubscripts shape;
    int rank{0};
    const ConstantSubscripts *shapes[sizeof...(TA)]{
        &std::get<I>(*args)->shape()...};
    const int ranks[sizeof...(TA)]{std::get<I>(*args)->Rank()...};
    for (unsigned int i{0}; i < sizeof...(TA); ++i) {
      if (ranks[i] > 0) {
        if (rank == 0) {
          rank = ranks[i];
          shape = *shapes[i];
        } else {
          if (shape != *shapes[i]) {
            // TODO: Rank compatibility was already checked but it seems to be
            // the first place where the actual shapes are checked to be the
            // same. Shouldn't this be checked elsewhere so that this is also
            // checked for non constexpr call to elemental intrinsics function?
            context.messages().Say(
                "Arguments in elemental intrinsic function are not conformable"_err_en_US);
            return Expr<TR>{std::move(funcRef)};
          }
        }
      }
    }
    CHECK(rank == GetRank(shape));

    // Compute all the scalar values of the results
    std::vector<Scalar<TR>> results;
    if (TotalElementCount(shape) > 0) {
      ConstantBounds bounds{shape};
      ConstantSubscripts index(rank, 1);
      do {
        if constexpr (std::is_same_v<WrapperType<TR, TA...>,
                          ScalarFuncWithContext<TR, TA...>>) {
          results.emplace_back(func(context,
              (ranks[I] ? std::get<I>(*args)->At(index)
                        : std::get<I>(*args)->GetScalarValue().value())...));
        } else if constexpr (std::is_same_v<WrapperType<TR, TA...>,
                                 ScalarFunc<TR, TA...>>) {
          results.emplace_back(func(
              (ranks[I] ? std::get<I>(*args)->At(index)
                        : std::get<I>(*args)->GetScalarValue().value())...));
        }
      } while (bounds.IncrementSubscripts(index));
    }
    // Build and return constant result
    if constexpr (TR::category == TypeCategory::Character) {
      auto len{static_cast<ConstantSubscript>(
          results.size() ? results[0].length() : 0)};
      return Expr<TR>{Constant<TR>{len, std::move(results), std::move(shape)}};
    } else {
      return Expr<TR>{Constant<TR>{std::move(results), std::move(shape)}};
    }
  }
  return Expr<TR>{std::move(funcRef)};
}

template <typename TR, typename... TA>
Expr<TR> FoldElementalIntrinsic(FoldingContext &context,
    FunctionRef<TR> &&funcRef, ScalarFunc<TR, TA...> func) {
  return FoldElementalIntrinsicHelper<ScalarFunc, TR, TA...>(
      context, std::move(funcRef), func, std::index_sequence_for<TA...>{});
}
template <typename TR, typename... TA>
Expr<TR> FoldElementalIntrinsic(FoldingContext &context,
    FunctionRef<TR> &&funcRef, ScalarFuncWithContext<TR, TA...> func) {
  return FoldElementalIntrinsicHelper<ScalarFuncWithContext, TR, TA...>(
      context, std::move(funcRef), func, std::index_sequence_for<TA...>{});
}

std::optional<std::int64_t> GetInt64Arg(const std::optional<ActualArgument> &);
std::optional<std::int64_t> GetInt64ArgOr(
    const std::optional<ActualArgument> &, std::int64_t defaultValue);

template <typename A, typename B>
std::optional<std::vector<A>> GetIntegerVector(const B &x) {
  static_assert(std::is_integral_v<A>);
  if (const auto *someInteger{UnwrapExpr<Expr<SomeInteger>>(x)}) {
    return std::visit(
        [](const auto &typedExpr) -> std::optional<std::vector<A>> {
          using T = ResultType<decltype(typedExpr)>;
          if (const auto *constant{UnwrapConstantValue<T>(typedExpr)}) {
            if (constant->Rank() == 1) {
              std::vector<A> result;
              for (const auto &value : constant->values()) {
                result.push_back(static_cast<A>(value.ToInt64()));
              }
              return result;
            }
          }
          return std::nullopt;
        },
        someInteger->u);
  }
  return std::nullopt;
}

// Transform an intrinsic function reference that contains user errors
// into an intrinsic with the same characteristic but the "invalid" name.
// This to prevent generating warnings over and over if the expression
// gets re-folded.
template <typename T> Expr<T> MakeInvalidIntrinsic(FunctionRef<T> &&funcRef) {
  SpecificIntrinsic invalid{std::get<SpecificIntrinsic>(funcRef.proc().u)};
  invalid.name = "(invalid intrinsic function call)";
  return Expr<T>{FunctionRef<T>{ProcedureDesignator{std::move(invalid)},
      ActualArguments{ActualArgument{AsGenericExpr(std::move(funcRef))}}}};
}

template <typename T> Expr<T> Folder<T>::Reshape(FunctionRef<T> &&funcRef) {
  auto args{funcRef.arguments()};
  CHECK(args.size() == 4);
  const auto *source{UnwrapConstantValue<T>(args[0])};
  const auto *pad{UnwrapConstantValue<T>(args[2])};
  std::optional<std::vector<ConstantSubscript>> shape{
      GetIntegerVector<ConstantSubscript>(args[1])};
  std::optional<std::vector<int>> order{GetIntegerVector<int>(args[3])};
  if (!source || !shape || (args[2] && !pad) || (args[3] && !order)) {
    return Expr<T>{std::move(funcRef)}; // Non-constant arguments
  } else if (!IsValidShape(shape.value())) {
    context_.messages().Say("Invalid SHAPE in RESHAPE"_en_US);
  } else {
    int rank{GetRank(shape.value())};
    std::size_t resultElements{TotalElementCount(shape.value())};
    std::optional<std::vector<int>> dimOrder;
    if (order) {
      dimOrder = ValidateDimensionOrder(rank, *order);
    }
    std::vector<int> *dimOrderPtr{dimOrder ? &dimOrder.value() : nullptr};
    if (order && !dimOrder) {
      context_.messages().Say("Invalid ORDER in RESHAPE"_en_US);
    } else if (resultElements > source->size() && (!pad || pad->empty())) {
      context_.messages().Say("Too few SOURCE elements in RESHAPE and PAD"
                              "is not present or has null size"_en_US);
    } else {
      Constant<T> result{!source->empty()
              ? source->Reshape(std::move(shape.value()))
              : pad->Reshape(std::move(shape.value()))};
      ConstantSubscripts subscripts{result.lbounds()};
      auto copied{result.CopyFrom(*source,
          std::min(source->size(), resultElements), subscripts, dimOrderPtr)};
      if (copied < resultElements) {
        CHECK(pad);
        copied += result.CopyFrom(
            *pad, resultElements - copied, subscripts, dimOrderPtr);
      }
      CHECK(copied == resultElements);
      return Expr<T>{std::move(result)};
    }
  }
  // Invalid, prevent re-folding
  return MakeInvalidIntrinsic(std::move(funcRef));
}

template <typename T>
Expr<T> FoldMINorMAX(
    FoldingContext &context, FunctionRef<T> &&funcRef, Ordering order) {
  std::vector<Constant<T> *> constantArgs;
  // Call Folding on all arguments, even if some are not constant,
  // to make operand promotion explicit.
  for (auto &arg : funcRef.arguments()) {
    if (auto *cst{Folder<T>{context}.Folding(arg)}) {
      constantArgs.push_back(cst);
    }
  }
  if (constantArgs.size() != funcRef.arguments().size())
    return Expr<T>(std::move(funcRef));
  CHECK(constantArgs.size() > 0);
  Expr<T> result{std::move(*constantArgs[0])};
  for (std::size_t i{1}; i < constantArgs.size(); ++i) {
    Extremum<T> extremum{order, result, Expr<T>{std::move(*constantArgs[i])}};
    result = FoldOperation(context, std::move(extremum));
  }
  return result;
}

// For AMAX0, AMIN0, AMAX1, AMIN1, DMAX1, DMIN1, MAX0, MIN0, MAX1, and MIN1
// a special care has to be taken to insert the conversion on the result
// of the MIN/MAX. This is made slightly more complex by the extension
// supported by f18 that arguments may have different kinds. This implies
// that the created MIN/MAX result type cannot be deduced from the standard but
// has to be deduced from the arguments.
// e.g. AMAX0(int8, int4) is rewritten to REAL(MAX(int8, INT(int4, 8)))).
template <typename T>
Expr<T> RewriteSpecificMINorMAX(
    FoldingContext &context, FunctionRef<T> &&funcRef) {
  ActualArguments &args{funcRef.arguments()};
  auto &intrinsic{DEREF(std::get_if<SpecificIntrinsic>(&funcRef.proc().u))};
  // Rewrite MAX1(args) to INT(MAX(args)) and fold. Same logic for MIN1.
  // Find result type for max/min based on the arguments.
  DynamicType resultType{args[0].value().GetType().value()};
  auto *resultTypeArg{&args[0]};
  for (auto j{args.size() - 1}; j > 0; --j) {
    DynamicType type{args[j].value().GetType().value()};
    if (type.category() == resultType.category()) {
      if (type.kind() > resultType.kind()) {
        resultTypeArg = &args[j];
        resultType = type;
      }
    } else if (resultType.category() == TypeCategory::Integer) {
      // Handle mixed real/integer arguments: all the previous arguments were
      // integers and this one is real. The type of the MAX/MIN result will
      // be the one of the real argument.
      resultTypeArg = &args[j];
      resultType = type;
    }
  }
  intrinsic.name =
      intrinsic.name.find("max") != std::string::npos ? "max"s : "min"s;
  intrinsic.characteristics.value().functionResult.value().SetType(resultType);
  auto insertConversion{[&](const auto &x) -> Expr<T> {
    using TR = ResultType<decltype(x)>;
    FunctionRef<TR> maxRef{std::move(funcRef.proc()), std::move(args)};
    return Fold(context, ConvertToType<T>(AsCategoryExpr(std::move(maxRef))));
  }};
  if (auto *sx{UnwrapExpr<Expr<SomeReal>>(*resultTypeArg)}) {
    return std::visit(insertConversion, sx->u);
  }
  auto &sx{DEREF(UnwrapExpr<Expr<SomeInteger>>(*resultTypeArg))};
  return std::visit(insertConversion, sx.u);
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, FunctionRef<T> &&funcRef) {
  ActualArguments &args{funcRef.arguments()};
  for (std::optional<ActualArgument> &arg : args) {
    if (auto *expr{UnwrapExpr<Expr<SomeType>>(arg)}) {
      *expr = Fold(context, std::move(*expr));
    }
  }
  if (auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)}) {
    const std::string name{intrinsic->name};
    if (name == "reshape") {
      return Folder<T>{context}.Reshape(std::move(funcRef));
    }
    // TODO: other type independent transformationals
    if constexpr (!std::is_same_v<T, SomeDerived>) {
      return FoldIntrinsicFunction(context, std::move(funcRef));
    }
  }
  return Expr<T>{std::move(funcRef)};
}

template <typename T>
Expr<T> FoldMerge(FoldingContext &context, FunctionRef<T> &&funcRef) {
  return FoldElementalIntrinsic<T, T, T, LogicalResult>(context,
      std::move(funcRef),
      ScalarFunc<T, T, T, LogicalResult>(
          [](const Scalar<T> &ifTrue, const Scalar<T> &ifFalse,
              const Scalar<LogicalResult> &predicate) -> Scalar<T> {
            return predicate.IsTrue() ? ifTrue : ifFalse;
          }));
}

Expr<ImpliedDoIndex::Result> FoldOperation(FoldingContext &, ImpliedDoIndex &&);

// Array constructor folding
template <typename T> class ArrayConstructorFolder {
public:
  explicit ArrayConstructorFolder(const FoldingContext &c) : context_{c} {}

  Expr<T> FoldArray(ArrayConstructor<T> &&array) {
    // Calls FoldArray(const ArrayConstructorValues<T> &) below
    if (FoldArray(array)) {
      auto n{static_cast<ConstantSubscript>(elements_.size())};
      if constexpr (std::is_same_v<T, SomeDerived>) {
        return Expr<T>{Constant<T>{array.GetType().GetDerivedTypeSpec(),
            std::move(elements_), ConstantSubscripts{n}}};
      } else if constexpr (T::category == TypeCategory::Character) {
        auto length{Fold(context_, common::Clone(array.LEN()))};
        if (std::optional<ConstantSubscript> lengthValue{ToInt64(length)}) {
          return Expr<T>{Constant<T>{
              *lengthValue, std::move(elements_), ConstantSubscripts{n}}};
        }
      } else {
        return Expr<T>{
            Constant<T>{std::move(elements_), ConstantSubscripts{n}}};
      }
    }
    return Expr<T>{std::move(array)};
  }

private:
  bool FoldArray(const common::CopyableIndirection<Expr<T>> &expr) {
    Expr<T> folded{Fold(context_, common::Clone(expr.value()))};
    if (const auto *c{UnwrapConstantValue<T>(folded)}) {
      // Copy elements in Fortran array element order
      ConstantSubscripts shape{c->shape()};
      int rank{c->Rank()};
      ConstantSubscripts index(GetRank(shape), 1);
      for (std::size_t n{c->size()}; n-- > 0;) {
        elements_.emplace_back(c->At(index));
        for (int d{0}; d < rank; ++d) {
          if (++index[d] <= shape[d]) {
            break;
          }
          index[d] = 1;
        }
      }
      return true;
    } else {
      return false;
    }
  }
  bool FoldArray(const ImpliedDo<T> &iDo) {
    Expr<SubscriptInteger> lower{
        Fold(context_, Expr<SubscriptInteger>{iDo.lower()})};
    Expr<SubscriptInteger> upper{
        Fold(context_, Expr<SubscriptInteger>{iDo.upper()})};
    Expr<SubscriptInteger> stride{
        Fold(context_, Expr<SubscriptInteger>{iDo.stride()})};
    std::optional<ConstantSubscript> start{ToInt64(lower)}, end{ToInt64(upper)},
        step{ToInt64(stride)};
    if (start && end && step && *step != 0) {
      bool result{true};
      ConstantSubscript &j{context_.StartImpliedDo(iDo.name(), *start)};
      if (*step > 0) {
        for (; j <= *end; j += *step) {
          result &= FoldArray(iDo.values());
        }
      } else {
        for (; j >= *end; j += *step) {
          result &= FoldArray(iDo.values());
        }
      }
      context_.EndImpliedDo(iDo.name());
      return result;
    } else {
      return false;
    }
  }
  bool FoldArray(const ArrayConstructorValue<T> &x) {
    return std::visit([&](const auto &y) { return FoldArray(y); }, x.u);
  }
  bool FoldArray(const ArrayConstructorValues<T> &xs) {
    for (const auto &x : xs) {
      if (!FoldArray(x)) {
        return false;
      }
    }
    return true;
  }

  FoldingContext context_;
  std::vector<Scalar<T>> elements_;
};

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, ArrayConstructor<T> &&array) {
  return ArrayConstructorFolder<T>{context}.FoldArray(std::move(array));
}

// Array operation elemental application: When all operands to an operation
// are constant arrays, array constructors without any implied DO loops,
// &/or expanded scalars, pull the operation "into" the array result by
// applying it in an elementwise fashion.  For example, [A,1]+[B,2]
// is rewritten into [A+B,1+2] and then partially folded to [A+B,3].

// If possible, restructures an array expression into an array constructor
// that comprises a "flat" ArrayConstructorValues with no implied DO loops.
template <typename T>
bool ArrayConstructorIsFlat(const ArrayConstructorValues<T> &values) {
  for (const ArrayConstructorValue<T> &x : values) {
    if (!std::holds_alternative<Expr<T>>(x.u)) {
      return false;
    }
  }
  return true;
}

template <typename T>
std::optional<Expr<T>> AsFlatArrayConstructor(const Expr<T> &expr) {
  if (const auto *c{UnwrapConstantValue<T>(expr)}) {
    ArrayConstructor<T> result{expr};
    if (c->size() > 0) {
      ConstantSubscripts at{c->lbounds()};
      do {
        result.Push(Expr<T>{Constant<T>{c->At(at)}});
      } while (c->IncrementSubscripts(at));
    }
    return std::make_optional<Expr<T>>(std::move(result));
  } else if (const auto *a{UnwrapExpr<ArrayConstructor<T>>(expr)}) {
    if (ArrayConstructorIsFlat(*a)) {
      return std::make_optional<Expr<T>>(expr);
    }
  } else if (const auto *p{UnwrapExpr<Parentheses<T>>(expr)}) {
    return AsFlatArrayConstructor(Expr<T>{p->left()});
  }
  return std::nullopt;
}

template <TypeCategory CAT>
std::enable_if_t<CAT != TypeCategory::Derived,
    std::optional<Expr<SomeKind<CAT>>>>
AsFlatArrayConstructor(const Expr<SomeKind<CAT>> &expr) {
  return std::visit(
      [&](const auto &kindExpr) -> std::optional<Expr<SomeKind<CAT>>> {
        if (auto flattened{AsFlatArrayConstructor(kindExpr)}) {
          return Expr<SomeKind<CAT>>{std::move(*flattened)};
        } else {
          return std::nullopt;
        }
      },
      expr.u);
}

// FromArrayConstructor is a subroutine for MapOperation() below.
// Given a flat ArrayConstructor<T> and a shape, it wraps the array
// into an Expr<T>, folds it, and returns the resulting wrapped
// array constructor or constant array value.
template <typename T>
Expr<T> FromArrayConstructor(FoldingContext &context,
    ArrayConstructor<T> &&values, std::optional<ConstantSubscripts> &&shape) {
  Expr<T> result{Fold(context, Expr<T>{std::move(values)})};
  if (shape) {
    if (auto *constant{UnwrapConstantValue<T>(result)}) {
      return Expr<T>{constant->Reshape(std::move(*shape))};
    }
  }
  return result;
}

// MapOperation is a utility for various specializations of ApplyElementwise()
// that follow.  Given one or two flat ArrayConstructor<OPERAND> (wrapped in an
// Expr<OPERAND>) for some specific operand type(s), apply a given function f
// to each of their corresponding elements to produce a flat
// ArrayConstructor<RESULT> (wrapped in an Expr<RESULT>).
// Preserves shape.

// Unary case
template <typename RESULT, typename OPERAND>
Expr<RESULT> MapOperation(FoldingContext &context,
    std::function<Expr<RESULT>(Expr<OPERAND> &&)> &&f, const Shape &shape,
    Expr<OPERAND> &&values) {
  ArrayConstructor<RESULT> result{values};
  if constexpr (common::HasMember<OPERAND, AllIntrinsicCategoryTypes>) {
    std::visit(
        [&](auto &&kindExpr) {
          using kindType = ResultType<decltype(kindExpr)>;
          auto &aConst{std::get<ArrayConstructor<kindType>>(kindExpr.u)};
          for (auto &acValue : aConst) {
            auto &scalar{std::get<Expr<kindType>>(acValue.u)};
            result.Push(Fold(context, f(Expr<OPERAND>{std::move(scalar)})));
          }
        },
        std::move(values.u));
  } else {
    auto &aConst{std::get<ArrayConstructor<OPERAND>>(values.u)};
    for (auto &acValue : aConst) {
      auto &scalar{std::get<Expr<OPERAND>>(acValue.u)};
      result.Push(Fold(context, f(std::move(scalar))));
    }
  }
  return FromArrayConstructor(
      context, std::move(result), AsConstantExtents(context, shape));
}

// array * array case
template <typename RESULT, typename LEFT, typename RIGHT>
Expr<RESULT> MapOperation(FoldingContext &context,
    std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)> &&f,
    const Shape &shape, Expr<LEFT> &&leftValues, Expr<RIGHT> &&rightValues) {
  ArrayConstructor<RESULT> result{leftValues};
  auto &leftArrConst{std::get<ArrayConstructor<LEFT>>(leftValues.u)};
  if constexpr (common::HasMember<RIGHT, AllIntrinsicCategoryTypes>) {
    std::visit(
        [&](auto &&kindExpr) {
          using kindType = ResultType<decltype(kindExpr)>;

          auto &rightArrConst{std::get<ArrayConstructor<kindType>>(kindExpr.u)};
          auto rightIter{rightArrConst.begin()};
          for (auto &leftValue : leftArrConst) {
            CHECK(rightIter != rightArrConst.end());
            auto &leftScalar{std::get<Expr<LEFT>>(leftValue.u)};
            auto &rightScalar{std::get<Expr<kindType>>(rightIter->u)};
            result.Push(Fold(context,
                f(std::move(leftScalar), Expr<RIGHT>{std::move(rightScalar)})));
            ++rightIter;
          }
        },
        std::move(rightValues.u));
  } else {
    auto &rightArrConst{std::get<ArrayConstructor<RIGHT>>(rightValues.u)};
    auto rightIter{rightArrConst.begin()};
    for (auto &leftValue : leftArrConst) {
      CHECK(rightIter != rightArrConst.end());
      auto &leftScalar{std::get<Expr<LEFT>>(leftValue.u)};
      auto &rightScalar{std::get<Expr<RIGHT>>(rightIter->u)};
      result.Push(
          Fold(context, f(std::move(leftScalar), std::move(rightScalar))));
      ++rightIter;
    }
  }
  return FromArrayConstructor(
      context, std::move(result), AsConstantExtents(context, shape));
}

// array * scalar case
template <typename RESULT, typename LEFT, typename RIGHT>
Expr<RESULT> MapOperation(FoldingContext &context,
    std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)> &&f,
    const Shape &shape, Expr<LEFT> &&leftValues,
    const Expr<RIGHT> &rightScalar) {
  ArrayConstructor<RESULT> result{leftValues};
  auto &leftArrConst{std::get<ArrayConstructor<LEFT>>(leftValues.u)};
  for (auto &leftValue : leftArrConst) {
    auto &leftScalar{std::get<Expr<LEFT>>(leftValue.u)};
    result.Push(
        Fold(context, f(std::move(leftScalar), Expr<RIGHT>{rightScalar})));
  }
  return FromArrayConstructor(
      context, std::move(result), AsConstantExtents(context, shape));
}

// scalar * array case
template <typename RESULT, typename LEFT, typename RIGHT>
Expr<RESULT> MapOperation(FoldingContext &context,
    std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)> &&f,
    const Shape &shape, const Expr<LEFT> &leftScalar,
    Expr<RIGHT> &&rightValues) {
  ArrayConstructor<RESULT> result{leftScalar};
  if constexpr (common::HasMember<RIGHT, AllIntrinsicCategoryTypes>) {
    std::visit(
        [&](auto &&kindExpr) {
          using kindType = ResultType<decltype(kindExpr)>;
          auto &rightArrConst{std::get<ArrayConstructor<kindType>>(kindExpr.u)};
          for (auto &rightValue : rightArrConst) {
            auto &rightScalar{std::get<Expr<kindType>>(rightValue.u)};
            result.Push(Fold(context,
                f(Expr<LEFT>{leftScalar},
                    Expr<RIGHT>{std::move(rightScalar)})));
          }
        },
        std::move(rightValues.u));
  } else {
    auto &rightArrConst{std::get<ArrayConstructor<RIGHT>>(rightValues.u)};
    for (auto &rightValue : rightArrConst) {
      auto &rightScalar{std::get<Expr<RIGHT>>(rightValue.u)};
      result.Push(
          Fold(context, f(Expr<LEFT>{leftScalar}, std::move(rightScalar))));
    }
  }
  return FromArrayConstructor(
      context, std::move(result), AsConstantExtents(context, shape));
}

// ApplyElementwise() recursively folds the operand expression(s) of an
// operation, then attempts to apply the operation to the (corresponding)
// scalar element(s) of those operands.  Returns std::nullopt for scalars
// or unlinearizable operands.
template <typename DERIVED, typename RESULT, typename OPERAND>
auto ApplyElementwise(FoldingContext &context,
    Operation<DERIVED, RESULT, OPERAND> &operation,
    std::function<Expr<RESULT>(Expr<OPERAND> &&)> &&f)
    -> std::optional<Expr<RESULT>> {
  auto &expr{operation.left()};
  expr = Fold(context, std::move(expr));
  if (expr.Rank() > 0) {
    if (std::optional<Shape> shape{GetShape(context, expr)}) {
      if (auto values{AsFlatArrayConstructor(expr)}) {
        return MapOperation(context, std::move(f), *shape, std::move(*values));
      }
    }
  }
  return std::nullopt;
}

template <typename DERIVED, typename RESULT, typename OPERAND>
auto ApplyElementwise(
    FoldingContext &context, Operation<DERIVED, RESULT, OPERAND> &operation)
    -> std::optional<Expr<RESULT>> {
  return ApplyElementwise(context, operation,
      std::function<Expr<RESULT>(Expr<OPERAND> &&)>{
          [](Expr<OPERAND> &&operand) {
            return Expr<RESULT>{DERIVED{std::move(operand)}};
          }});
}

// Predicate: is a scalar expression suitable for naive scalar expansion
// in the flattening of an array expression?
// TODO: capture such scalar expansions in temporaries, flatten everything
struct UnexpandabilityFindingVisitor
    : public AnyTraverse<UnexpandabilityFindingVisitor> {
  using Base = AnyTraverse<UnexpandabilityFindingVisitor>;
  using Base::operator();
  UnexpandabilityFindingVisitor() : Base{*this} {}
  template <typename T> bool operator()(const FunctionRef<T> &) { return true; }
  bool operator()(const CoarrayRef &) { return true; }
};

template <typename T> bool IsExpandableScalar(const Expr<T> &expr) {
  return !UnexpandabilityFindingVisitor{}(expr);
}

template <typename DERIVED, typename RESULT, typename LEFT, typename RIGHT>
auto ApplyElementwise(FoldingContext &context,
    Operation<DERIVED, RESULT, LEFT, RIGHT> &operation,
    std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)> &&f)
    -> std::optional<Expr<RESULT>> {
  auto &leftExpr{operation.left()};
  leftExpr = Fold(context, std::move(leftExpr));
  auto &rightExpr{operation.right()};
  rightExpr = Fold(context, std::move(rightExpr));
  if (leftExpr.Rank() > 0) {
    if (std::optional<Shape> leftShape{GetShape(context, leftExpr)}) {
      if (auto left{AsFlatArrayConstructor(leftExpr)}) {
        if (rightExpr.Rank() > 0) {
          if (std::optional<Shape> rightShape{GetShape(context, rightExpr)}) {
            if (auto right{AsFlatArrayConstructor(rightExpr)}) {
              CheckConformance(context.messages(), *leftShape, *rightShape);
              return MapOperation(context, std::move(f), *leftShape,
                  std::move(*left), std::move(*right));
            }
          }
        } else if (IsExpandableScalar(rightExpr)) {
          return MapOperation(
              context, std::move(f), *leftShape, std::move(*left), rightExpr);
        }
      }
    }
  } else if (rightExpr.Rank() > 0 && IsExpandableScalar(leftExpr)) {
    if (std::optional<Shape> shape{GetShape(context, rightExpr)}) {
      if (auto right{AsFlatArrayConstructor(rightExpr)}) {
        return MapOperation(
            context, std::move(f), *shape, leftExpr, std::move(*right));
      }
    }
  }
  return std::nullopt;
}

template <typename DERIVED, typename RESULT, typename LEFT, typename RIGHT>
auto ApplyElementwise(
    FoldingContext &context, Operation<DERIVED, RESULT, LEFT, RIGHT> &operation)
    -> std::optional<Expr<RESULT>> {
  return ApplyElementwise(context, operation,
      std::function<Expr<RESULT>(Expr<LEFT> &&, Expr<RIGHT> &&)>{
          [](Expr<LEFT> &&left, Expr<RIGHT> &&right) {
            return Expr<RESULT>{DERIVED{std::move(left), std::move(right)}};
          }});
}

// Unary operations

template <typename TO, typename FROM>
common::IfNoLvalue<std::optional<TO>, FROM> ConvertString(FROM &&s) {
  if constexpr (std::is_same_v<TO, FROM>) {
    return std::make_optional<TO>(std::move(s));
  } else {
    // Fortran character conversion is well defined between distinct kinds
    // only when the actual characters are valid 7-bit ASCII.
    TO str;
    for (auto iter{s.cbegin()}; iter != s.cend(); ++iter) {
      if (static_cast<std::uint64_t>(*iter) > 127) {
        return std::nullopt;
      }
      str.push_back(*iter);
    }
    return std::make_optional<TO>(std::move(str));
  }
}

template <typename TO, TypeCategory FROMCAT>
Expr<TO> FoldOperation(
    FoldingContext &context, Convert<TO, FROMCAT> &&convert) {
  if (auto array{ApplyElementwise(context, convert)}) {
    return *array;
  }
  return std::visit(
      [&](auto &kindExpr) -> Expr<TO> {
        using Operand = ResultType<decltype(kindExpr)>;
        char buffer[64];
        if (auto value{GetScalarConstantValue<Operand>(kindExpr)}) {
          if constexpr (TO::category == TypeCategory::Integer) {
            if constexpr (Operand::category == TypeCategory::Integer) {
              auto converted{Scalar<TO>::ConvertSigned(*value)};
              if (converted.overflow) {
                context.messages().Say(
                    "INTEGER(%d) to INTEGER(%d) conversion overflowed"_en_US,
                    Operand::kind, TO::kind);
              }
              return ScalarConstantToExpr(std::move(converted.value));
            } else if constexpr (Operand::category == TypeCategory::Real) {
              auto converted{value->template ToInteger<Scalar<TO>>()};
              if (converted.flags.test(RealFlag::InvalidArgument)) {
                context.messages().Say(
                    "REAL(%d) to INTEGER(%d) conversion: invalid argument"_en_US,
                    Operand::kind, TO::kind);
              } else if (converted.flags.test(RealFlag::Overflow)) {
                context.messages().Say(
                    "REAL(%d) to INTEGER(%d) conversion overflowed"_en_US,
                    Operand::kind, TO::kind);
              }
              return ScalarConstantToExpr(std::move(converted.value));
            }
          } else if constexpr (TO::category == TypeCategory::Real) {
            if constexpr (Operand::category == TypeCategory::Integer) {
              auto converted{Scalar<TO>::FromInteger(*value)};
              if (!converted.flags.empty()) {
                std::snprintf(buffer, sizeof buffer,
                    "INTEGER(%d) to REAL(%d) conversion", Operand::kind,
                    TO::kind);
                RealFlagWarnings(context, converted.flags, buffer);
              }
              return ScalarConstantToExpr(std::move(converted.value));
            } else if constexpr (Operand::category == TypeCategory::Real) {
              auto converted{Scalar<TO>::Convert(*value)};
              if (!converted.flags.empty()) {
                std::snprintf(buffer, sizeof buffer,
                    "REAL(%d) to REAL(%d) conversion", Operand::kind, TO::kind);
                RealFlagWarnings(context, converted.flags, buffer);
              }
              if (context.flushSubnormalsToZero()) {
                converted.value = converted.value.FlushSubnormalToZero();
              }
              return ScalarConstantToExpr(std::move(converted.value));
            }
          } else if constexpr (TO::category == TypeCategory::Character &&
              Operand::category == TypeCategory::Character) {
            if (auto converted{ConvertString<Scalar<TO>>(std::move(*value))}) {
              return ScalarConstantToExpr(std::move(*converted));
            }
          } else if constexpr (TO::category == TypeCategory::Logical &&
              Operand::category == TypeCategory::Logical) {
            return Expr<TO>{value->IsTrue()};
          }
        } else if constexpr (std::is_same_v<Operand, TO> &&
            FROMCAT != TypeCategory::Character) {
          return std::move(kindExpr); // remove needless conversion
        }
        return Expr<TO>{std::move(convert)};
      },
      convert.left().u);
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, Parentheses<T> &&x) {
  auto &operand{x.left()};
  operand = Fold(context, std::move(operand));
  if (auto value{GetScalarConstantValue<T>(operand)}) {
    // Preserve parentheses, even around constants.
    return Expr<T>{Parentheses<T>{Expr<T>{Constant<T>{*value}}}};
  } else if (std::holds_alternative<Parentheses<T>>(operand.u)) {
    // ((x)) -> (x)
    return std::move(operand);
  } else {
    return Expr<T>{Parentheses<T>{std::move(operand)}};
  }
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, Negate<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  auto &operand{x.left()};
  if (auto value{GetScalarConstantValue<T>(operand)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto negated{value->Negate()};
      if (negated.overflow) {
        context.messages().Say(
            "INTEGER(%d) negation overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{std::move(negated.value)}};
    } else {
      // REAL & COMPLEX negation: no exceptions possible
      return Expr<T>{Constant<T>{value->Negate()}};
    }
  }
  return Expr<T>{std::move(x)};
}

// Binary (dyadic) operations

template <typename LEFT, typename RIGHT>
std::optional<std::pair<Scalar<LEFT>, Scalar<RIGHT>>> OperandsAreConstants(
    const Expr<LEFT> &x, const Expr<RIGHT> &y) {
  if (auto xvalue{GetScalarConstantValue<LEFT>(x)}) {
    if (auto yvalue{GetScalarConstantValue<RIGHT>(y)}) {
      return {std::make_pair(*xvalue, *yvalue)};
    }
  }
  return std::nullopt;
}

template <typename DERIVED, typename RESULT, typename LEFT, typename RIGHT>
std::optional<std::pair<Scalar<LEFT>, Scalar<RIGHT>>> OperandsAreConstants(
    const Operation<DERIVED, RESULT, LEFT, RIGHT> &operation) {
  return OperandsAreConstants(operation.left(), operation.right());
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, Add<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto sum{folded->first.AddSigned(folded->second)};
      if (sum.overflow) {
        context.messages().Say(
            "INTEGER(%d) addition overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{sum.value}};
    } else {
      auto sum{folded->first.Add(folded->second, context.rounding())};
      RealFlagWarnings(context, sum.flags, "addition");
      if (context.flushSubnormalsToZero()) {
        sum.value = sum.value.FlushSubnormalToZero();
      }
      return Expr<T>{Constant<T>{sum.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, Subtract<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto difference{folded->first.SubtractSigned(folded->second)};
      if (difference.overflow) {
        context.messages().Say(
            "INTEGER(%d) subtraction overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{difference.value}};
    } else {
      auto difference{
          folded->first.Subtract(folded->second, context.rounding())};
      RealFlagWarnings(context, difference.flags, "subtraction");
      if (context.flushSubnormalsToZero()) {
        difference.value = difference.value.FlushSubnormalToZero();
      }
      return Expr<T>{Constant<T>{difference.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, Multiply<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto product{folded->first.MultiplySigned(folded->second)};
      if (product.SignedMultiplicationOverflowed()) {
        context.messages().Say(
            "INTEGER(%d) multiplication overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{product.lower}};
    } else {
      auto product{folded->first.Multiply(folded->second, context.rounding())};
      RealFlagWarnings(context, product.flags, "multiplication");
      if (context.flushSubnormalsToZero()) {
        product.value = product.value.FlushSubnormalToZero();
      }
      return Expr<T>{Constant<T>{product.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, Divide<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto quotAndRem{folded->first.DivideSigned(folded->second)};
      if (quotAndRem.divisionByZero) {
        context.messages().Say("INTEGER(%d) division by zero"_en_US, T::kind);
        return Expr<T>{std::move(x)};
      }
      if (quotAndRem.overflow) {
        context.messages().Say(
            "INTEGER(%d) division overflowed"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{quotAndRem.quotient}};
    } else {
      auto quotient{folded->first.Divide(folded->second, context.rounding())};
      RealFlagWarnings(context, quotient.flags, "division");
      if (context.flushSubnormalsToZero()) {
        quotient.value = quotient.value.FlushSubnormalToZero();
      }
      return Expr<T>{Constant<T>{quotient.value}};
    }
  }
  return Expr<T>{std::move(x)};
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, Power<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto power{folded->first.Power(folded->second)};
      if (power.divisionByZero) {
        context.messages().Say(
            "INTEGER(%d) zero to negative power"_en_US, T::kind);
      } else if (power.overflow) {
        context.messages().Say("INTEGER(%d) power overflowed"_en_US, T::kind);
      } else if (power.zeroToZero) {
        context.messages().Say(
            "INTEGER(%d) 0**0 is not defined"_en_US, T::kind);
      }
      return Expr<T>{Constant<T>{power.power}};
    } else {
      if (auto callable{context.hostIntrinsicsLibrary()
                            .GetHostProcedureWrapper<Scalar, T, T, T>("pow")}) {
        return Expr<T>{
            Constant<T>{(*callable)(context, folded->first, folded->second)}};
      } else {
        context.messages().Say(
            "Power for %s cannot be folded on host"_en_US, T{}.AsFortran());
      }
    }
  }
  return Expr<T>{std::move(x)};
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, RealToIntPower<T> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  return std::visit(
      [&](auto &y) -> Expr<T> {
        if (auto folded{OperandsAreConstants(x.left(), y)}) {
          auto power{evaluate::IntPower(folded->first, folded->second)};
          RealFlagWarnings(context, power.flags, "power with INTEGER exponent");
          if (context.flushSubnormalsToZero()) {
            power.value = power.value.FlushSubnormalToZero();
          }
          return Expr<T>{Constant<T>{power.value}};
        } else {
          return Expr<T>{std::move(x)};
        }
      },
      x.right().u);
}

template <typename T>
Expr<T> FoldOperation(FoldingContext &context, Extremum<T> &&x) {
  if (auto array{ApplyElementwise(context, x,
          std::function<Expr<T>(Expr<T> &&, Expr<T> &&)>{[=](Expr<T> &&l,
                                                             Expr<T> &&r) {
            return Expr<T>{Extremum<T>{x.ordering, std::move(l), std::move(r)}};
          }})}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(x)}) {
    if constexpr (T::category == TypeCategory::Integer) {
      if (folded->first.CompareSigned(folded->second) == x.ordering) {
        return Expr<T>{Constant<T>{folded->first}};
      }
    } else if constexpr (T::category == TypeCategory::Real) {
      if (folded->first.IsNotANumber() ||
          (folded->first.Compare(folded->second) == Relation::Less) ==
              (x.ordering == Ordering::Less)) {
        return Expr<T>{Constant<T>{folded->first}};
      }
    } else {
      static_assert(T::category == TypeCategory::Character);
      // Result of MIN and MAX on character has the length of
      // the longest argument.
      auto maxLen{std::max(folded->first.length(), folded->second.length())};
      bool isFirst{x.ordering == Compare(folded->first, folded->second)};
      auto res{isFirst ? std::move(folded->first) : std::move(folded->second)};
      res = res.length() == maxLen
          ? std::move(res)
          : CharacterUtils<T::kind>::Resize(res, maxLen);
      return Expr<T>{Constant<T>{std::move(res)}};
    }
    return Expr<T>{Constant<T>{folded->second}};
  }
  return Expr<T>{std::move(x)};
}

template <int KIND>
Expr<Type<TypeCategory::Real, KIND>> ToReal(
    FoldingContext &context, Expr<SomeType> &&expr) {
  using Result = Type<TypeCategory::Real, KIND>;
  std::optional<Expr<Result>> result;
  std::visit(
      [&](auto &&x) {
        using From = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<From, BOZLiteralConstant>) {
          // Move the bits without any integer->real conversion
          From original{x};
          result = ConvertToType<Result>(std::move(x));
          const auto *constant{UnwrapExpr<Constant<Result>>(*result)};
          CHECK(constant);
          Scalar<Result> real{constant->GetScalarValue().value()};
          From converted{From::ConvertUnsigned(real.RawBits()).value};
          if (original != converted) { // C1601
            context.messages().Say(
                "Nonzero bits truncated from BOZ literal constant in REAL intrinsic"_en_US);
          }
        } else if constexpr (IsNumericCategoryExpr<From>()) {
          result = Fold(context, ConvertToType<Result>(std::move(x)));
        } else {
          common::die("ToReal: bad argument expression");
        }
      },
      std::move(expr.u));
  return result.value();
}

template <typename T>
Expr<T> ExpressionBase<T>::Rewrite(FoldingContext &context, Expr<T> &&expr) {
  return std::visit(
      [&](auto &&x) -> Expr<T> {
        if constexpr (IsSpecificIntrinsicType<T>) {
          return FoldOperation(context, std::move(x));
        } else if constexpr (std::is_same_v<T, SomeDerived>) {
          return FoldOperation(context, std::move(x));
        } else if constexpr (common::HasMember<decltype(x),
                                 TypelessExpression>) {
          return std::move(expr);
        } else {
          return Expr<T>{Fold(context, std::move(x))};
        }
      },
      std::move(expr.u));
}

FOR_EACH_TYPE_AND_KIND(extern template class ExpressionBase, )

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_FOLD_IMPLEMENTATION_H_
