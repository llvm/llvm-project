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

#ifndef FORTRAN_LOWER_RTBUILDER_H
#define FORTRAN_LOWER_RTBUILDER_H

#include "flang/Lower/ConvertType.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

// List the runtime headers we want to be able to dissect
#include "../../runtime/io-api.h"

namespace Fortran::lower {

using TypeBuilderFunc = mlir::Type (*)(mlir::MLIRContext *);
using FuncTypeBuilderFunc = mlir::FunctionType (*)(mlir::MLIRContext *);

//===----------------------------------------------------------------------===//
// Type builder models
//===----------------------------------------------------------------------===//

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
constexpr TypeBuilderFunc getModel<Fortran::runtime::io::Iostat>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context,
                                  8 * sizeof(Fortran::runtime::io::Iostat));
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
constexpr TypeBuilderFunc getModel<void **>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(
        fir::PointerType::get(mlir::IntegerType::get(context, 8)));
  };
}
template <>
constexpr TypeBuilderFunc getModel<std::int64_t>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 64);
  };
}
template <>
constexpr TypeBuilderFunc getModel<std::int64_t &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    TypeBuilderFunc f{getModel<std::int64_t>()};
    return fir::ReferenceType::get(f(context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<std::size_t>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(context, 8 * sizeof(std::size_t));
  };
}
template <>
constexpr TypeBuilderFunc getModel<Fortran::runtime::io::IoStatementState *>() {
  return getModel<char *>();
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
constexpr TypeBuilderFunc getModel<const Fortran::runtime::Descriptor &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::BoxType::get(mlir::NoneType::get(context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<const Fortran::runtime::NamelistGroup &>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    // FIXME: a namelist group must be some well-defined data structure, use a
    // tuple as a proxy for the moment
    return mlir::TupleType::get(context);
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
#define AsSequence(X) decltype(Fortran::lower::details::filter(FullSeq(X){}))
#define mkKey(X)                                                               \
  Fortran::lower::RuntimeTableEntry<                                           \
      Fortran::lower::RuntimeTableKey<decltype(X)>, AsSequence(X)>

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_RTBUILDER_H
