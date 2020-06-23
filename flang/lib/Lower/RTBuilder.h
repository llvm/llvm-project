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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
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
    return mlir::IntegerType::get(8 * sizeof(int), context);
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
    return mlir::IntegerType::get(8 * sizeof(Fortran::runtime::io::Iostat),
                                  context);
  };
}
template <>
constexpr TypeBuilderFunc getModel<char *>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(mlir::IntegerType::get(8, context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<const char *>() {
  return getModel<char *>();
}
template <>
constexpr TypeBuilderFunc getModel<const char16_t *>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(mlir::IntegerType::get(16, context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<const char32_t *>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(mlir::IntegerType::get(32, context));
  };
}
template <>
constexpr TypeBuilderFunc getModel<void **>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::ReferenceType::get(
        fir::PointerType::get(mlir::IntegerType::get(8, context)));
  };
}
template <>
constexpr TypeBuilderFunc getModel<std::int64_t>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return mlir::IntegerType::get(64, context);
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
    return mlir::IntegerType::get(8 * sizeof(std::size_t), context);
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
    return mlir::IntegerType::get(1, context);
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
constexpr TypeBuilderFunc getModel<float _Complex>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::CplxType::get(context, sizeof(float));
  };
}
template <>
constexpr TypeBuilderFunc getModel<double _Complex>() {
  return [](mlir::MLIRContext *context) -> mlir::Type {
    return fir::CplxType::get(context, sizeof(double));
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
    return mlir::TupleType::get(llvm::None, context);
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
      return mlir::FunctionType::get(argTys, {retTy}, ctxt);
    };
  }
};

//===----------------------------------------------------------------------===//
// Runtime table building (constexpr folded)
//===----------------------------------------------------------------------===//

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-string-literal-operator-template"
#endif

// clang++ generates warnings about usage of a GNU extension, ignore them
template <char... Cs>
using RuntimeIdentifier = std::integer_sequence<char, Cs...>;
template <typename T, T... Cs>
static constexpr RuntimeIdentifier<Cs...> operator""_rt_ident() {
  return {};
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

template <typename...>
struct RuntimeTableEntry;
template <typename KT, char... Cs>
struct RuntimeTableEntry<RuntimeTableKey<KT>, RuntimeIdentifier<Cs...>> {
  static constexpr FuncTypeBuilderFunc getTypeModel() {
    return RuntimeTableKey<KT>::getTypeModel();
  }
  static constexpr const char name[sizeof...(Cs) + 1] = {Cs..., '\0'};
};

#define QuoteKey(X) #X##_rt_ident
#define ExpandKey(X) QuoteKey(X)
#define mkKey(X)                                                               \
  Fortran::lower::RuntimeTableEntry<                                           \
      Fortran::lower::RuntimeTableKey<decltype(X)>, decltype(ExpandKey(X))>

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_RTBUILDER_H
