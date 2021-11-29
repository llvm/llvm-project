//===-- RTBuilder.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines some C++17 template classes that are used to convert the
/// signatures of plain old C functions into a model that can be used to
/// generate MLIR calls to those functions. This can be used to autogenerate
/// tables at compiler compile-time to call runtime support code.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_RTBUILDER_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_RTBUILDER_H

#include "flang/Common/Fortran.h"
#include "flang/Common/uint128.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

// Incomplete type indicating C99 complex ABI in interfaces. Beware, _Complex
// and std::complex are layout compatible, but not compatible in all ABI call
// interfaces (e.g. X86 32 bits). _Complex is not standard C++, so do not use
// it here.
struct c_float_complex_t;
struct c_double_complex_t;

namespace Fortran::runtime {
class Descriptor;
}

namespace fir::runtime {

using TypeBuilderFunc = mlir::Type (*)(mlir::MLIRContext *);
using FuncTypeBuilderFunc = mlir::FunctionType (*)(mlir::MLIRContext *);

//===----------------------------------------------------------------------===//
// Type builder models
//===----------------------------------------------------------------------===//

// TODO: all usages of sizeof in this file assume build ==  host == target.
// This will need to be re-visited for cross compilation.

/// Return a function that returns the type signature model for the type `T`
/// when provided an MLIRContext*. This allows one to translate C(++) function
/// signatures from runtime header files to MLIR signatures into a static table
/// at compile-time.
///
/// For example, when `T` is `int`, return a function that returns the MLIR
/// standard type `i32` when `sizeof(int)` is 4.
template <typename T>
static constexpr TypeBuilderFunc getModel();
template <>
constexpr TypeBuilderFunc getModel<short int>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 8 * sizeof(short int));
  };
}
template <>
constexpr TypeBuilderFunc getModel<int>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 8 * sizeof(int));
  };
}
template <>
constexpr TypeBuilderFunc getModel<int &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    TypeBuilderFunc f{getModel<int>()};
    return fir::ReferenceType::get(f(context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<char *>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(mlir::IntegerType::get(context, 8));
  };
}
template <>
constexpr TypeBuilderFunc getModel<const char *>() {
  return getModel<char *>();
}
template <>
constexpr TypeBuilderFunc getModel<const char16_t *>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(mlir::IntegerType::get(context, 16));
  };
}
template <>
constexpr TypeBuilderFunc getModel<const char32_t *>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(mlir::IntegerType::get(context, 32));
  };
}
template <>
constexpr TypeBuilderFunc getModel<signed char>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 8 * sizeof(signed char));
  };
}
template <>
constexpr TypeBuilderFunc getModel<void *>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::LLVMPointerType::get(context,
                                     mlir::IntegerType::get(context, 8));
  };
}
template <>
constexpr TypeBuilderFunc getModel<void **>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(
        fir::LLVMPointerType::get(context, mlir::IntegerType::get(context, 8)));
  };
}
template <>
constexpr TypeBuilderFunc getModel<long>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 8 * sizeof(long));
  };
}
template <>
constexpr TypeBuilderFunc getModel<long &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    TypeBuilderFunc f{getModel<long>()};
    return fir::ReferenceType::get(f(context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<long *>() {
  return getModel<long &>();
}
template <>
constexpr TypeBuilderFunc getModel<long long>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 8 * sizeof(long long));
  };
}
template <>
constexpr TypeBuilderFunc getModel<Fortran::common::int128_t>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context,
                                  8 * sizeof(Fortran::common::int128_t));
  };
}
template <>
constexpr TypeBuilderFunc getModel<long long &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    TypeBuilderFunc f{getModel<long long>()};
    return fir::ReferenceType::get(f(context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<long long *>() {
  return getModel<long long &>();
}
template <>
constexpr TypeBuilderFunc getModel<unsigned long>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 8 * sizeof(unsigned long));
  };
}
template <>
constexpr TypeBuilderFunc getModel<unsigned long long>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 8 * sizeof(unsigned long long));
  };
}
template <>
constexpr TypeBuilderFunc getModel<double>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::FloatType::getF64(context);
  };
}
template <>
constexpr TypeBuilderFunc getModel<double &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    TypeBuilderFunc f{getModel<double>()};
    return fir::ReferenceType::get(f(context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<double *>() {
  return getModel<double &>();
}
template <>
constexpr TypeBuilderFunc getModel<float>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::FloatType::getF32(context);
  };
}
template <>
constexpr TypeBuilderFunc getModel<float &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    TypeBuilderFunc f{getModel<float>()};
    return fir::ReferenceType::get(f(context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<float *>() {
  return getModel<float &>();
}
template <>
constexpr TypeBuilderFunc getModel<bool>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 1);
  };
}
template <>
constexpr TypeBuilderFunc getModel<bool &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    TypeBuilderFunc f{getModel<bool>()};
    return fir::ReferenceType::get(f(context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<std::complex<float> &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    auto ty = mlir::ComplexType::get(mlir::FloatType::getF32(context));
    return fir::ReferenceType::get(ty);
  };
}
template <>
constexpr TypeBuilderFunc getModel<std::complex<double> &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    auto ty = mlir::ComplexType::get(mlir::FloatType::getF64(context));
    return fir::ReferenceType::get(ty);
  };
}
template <>
constexpr TypeBuilderFunc getModel<c_float_complex_t>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ComplexType::get(context, sizeof(float));
  };
}
template <>
constexpr TypeBuilderFunc getModel<c_double_complex_t>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ComplexType::get(context, sizeof(double));
  };
}
template <>
constexpr TypeBuilderFunc getModel<const Fortran::runtime::Descriptor &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::BoxType::get(mlir::NoneType::get(context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<Fortran::runtime::Descriptor &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(
        fir::BoxType::get(mlir::NoneType::get(context)));
  };
}
template <>
constexpr TypeBuilderFunc getModel<const Fortran::runtime::Descriptor *>() {
  return getModel<const Fortran::runtime::Descriptor &>();
}
template <>
constexpr TypeBuilderFunc getModel<Fortran::runtime::Descriptor *>() {
  return getModel<Fortran::runtime::Descriptor &>();
}
template <>
constexpr TypeBuilderFunc getModel<Fortran::common::TypeCategory>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context,
                                  sizeof(Fortran::common::TypeCategory) * 8);
  };
}
template <>
constexpr TypeBuilderFunc getModel<void>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::NoneType::get(context);
  };
}

template <typename...>
struct RuntimeTableKey;
template <typename RT, typename... ATs>
struct RuntimeTableKey<RT(ATs...)> {
  static constexpr FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctxt) {
      TypeBuilderFunc ret = getModel<RT>();
      std::array<TypeBuilderFunc, sizeof...(ATs)> args = {getModel<ATs>()...};
      mlir::Type retTy = ret(ctxt);
      llvm::SmallVector<mlir::Type, sizeof...(ATs)> argTys;
      for (auto f : args)
        argTys.push_back(f(ctxt));
      return mlir::FunctionType::get(ctxt, argTys, {retTy});
    };
  }
};

//===----------------------------------------------------------------------===//
// Runtime table building (constexpr folded)
//===----------------------------------------------------------------------===//

template <char... Cs>
using RuntimeIdentifier = std::integer_sequence<char, Cs...>;

namespace details {
template <typename T, T... As, T... Bs>
static constexpr std::integer_sequence<T, As..., Bs...>
concat(std::integer_sequence<T, As...>, std::integer_sequence<T, Bs...>) {
  return {};
}
template <typename T, T... As, T... Bs, typename... Cs>
static constexpr auto concat(std::integer_sequence<T, As...>,
                             std::integer_sequence<T, Bs...>, Cs...) {
  return concat(std::integer_sequence<T, As..., Bs...>{}, Cs{}...);
}
template <typename T>
static constexpr std::integer_sequence<T> concat(std::integer_sequence<T>) {
  return {};
}
template <typename T, T a>
static constexpr auto filterZero(std::integer_sequence<T, a>) {
  if constexpr (a != 0) {
    return std::integer_sequence<T, a>{};
  } else {
    return std::integer_sequence<T>{};
  }
}
template <typename T, T... b>
static constexpr auto filter(std::integer_sequence<T, b...>) {
  if constexpr (sizeof...(b) > 0) {
    return details::concat(filterZero(std::integer_sequence<T, b>{})...);
  } else {
    return std::integer_sequence<T>{};
  }
}
} // namespace details

template <typename...>
struct RuntimeTableEntry;
template <typename KT, char... Cs>
struct RuntimeTableEntry<RuntimeTableKey<KT>, RuntimeIdentifier<Cs...>> {
  static constexpr FuncTypeBuilderFunc getTypeModel() {
    return RuntimeTableKey<KT>::getTypeModel();
  }
  static constexpr const char name[sizeof...(Cs) + 1] = {Cs..., '\0'};
};

#undef E
#define E(L, I) (I < sizeof(L) / sizeof(*L) ? L[I] : 0)
#define QuoteKey(X) #X
#define ExpandAndQuoteKey(X) QuoteKey(X)
#define MacroExpandKey(X)                                                      \
  E(X, 0), E(X, 1), E(X, 2), E(X, 3), E(X, 4), E(X, 5), E(X, 6), E(X, 7),      \
      E(X, 8), E(X, 9), E(X, 10), E(X, 11), E(X, 12), E(X, 13), E(X, 14),      \
      E(X, 15), E(X, 16), E(X, 17), E(X, 18), E(X, 19), E(X, 20), E(X, 21),    \
      E(X, 22), E(X, 23), E(X, 24), E(X, 25), E(X, 26), E(X, 27), E(X, 28),    \
      E(X, 29), E(X, 30), E(X, 31), E(X, 32), E(X, 33), E(X, 34), E(X, 35),    \
      E(X, 36), E(X, 37), E(X, 38), E(X, 39), E(X, 40), E(X, 41), E(X, 42),    \
      E(X, 43), E(X, 44), E(X, 45), E(X, 46), E(X, 47), E(X, 48), E(X, 49)
#define ExpandKey(X) MacroExpandKey(QuoteKey(X))
#define FullSeq(X) std::integer_sequence<char, ExpandKey(X)>
#define AsSequence(X) decltype(fir::runtime::details::filter(FullSeq(X){}))
#define mkKey(X)                                                               \
  fir::runtime::RuntimeTableEntry<fir::runtime::RuntimeTableKey<decltype(X)>,  \
                                  AsSequence(X)>
#define mkRTKey(X) mkKey(RTNAME(X))

/// Get (or generate) the MLIR FuncOp for a given runtime function. Its template
/// argument is intended to be of the form: <mkRTKey(runtime function name)>
/// Clients should add "using namespace Fortran::runtime"
/// in order to use this function.
template <typename RuntimeEntry>
static mlir::FuncOp getRuntimeFunc(mlir::Location loc,
                                   fir::FirOpBuilder &builder) {
  auto name = RuntimeEntry::name;
  auto func = builder.getNamedFunction(name);
  if (func)
    return func;
  auto funTy = RuntimeEntry::getTypeModel()(builder.getContext());
  func = builder.createFunction(loc, name, funTy);
  func->setAttr("fir.runtime", builder.getUnitAttr());
  return func;
}

namespace helper {
template <int N, typename A>
void createArguments(llvm::SmallVectorImpl<mlir::Value> &result,
                     fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::FunctionType fTy, A arg) {
  result.emplace_back(builder.createConvert(loc, fTy.getInput(N), arg));
}

template <int N, typename A, typename... As>
void createArguments(llvm::SmallVectorImpl<mlir::Value> &result,
                     fir::FirOpBuilder &builder, mlir::Location loc,
                     mlir::FunctionType fTy, A arg, As... args) {
  result.emplace_back(builder.createConvert(loc, fTy.getInput(N), arg));
  createArguments<N + 1>(result, builder, loc, fTy, args...);
}
} // namespace helper

/// Create a SmallVector of arguments for a runtime call.
template <typename... As>
llvm::SmallVector<mlir::Value>
createArguments(fir::FirOpBuilder &builder, mlir::Location loc,
                mlir::FunctionType fTy, As... args) {
  llvm::SmallVector<mlir::Value> result;
  helper::createArguments<0>(result, builder, loc, fTy, args...);
  return result;
}

} // namespace fir::runtime

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_RTBUILDER_H
