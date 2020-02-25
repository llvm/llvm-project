//===-- include/flang/Evaluate/intrinsics-library.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_INTRINSICS_LIBRARY_H_
#define FORTRAN_EVALUATE_INTRINSICS_LIBRARY_H_

// Defines structures to be used in F18 for folding intrinsic function with host
// runtime libraries. To avoid unnecessary header circular dependencies, the
// actual implementation of the templatized member function are defined in
// intrinsics-library-templates.h The header at hand is meant to be included by
// files that need to define intrinsic runtime data structure but that do not
// use them directly. To actually use the runtime data structures,
// intrinsics-library-templates.h must be included.

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace Fortran::evaluate {
class FoldingContext;

using TypeCode = unsigned char;

template<typename TR, typename... TA> using FuncPointer = TR (*)(TA...);
// This specific type signature prevents GCC complaining about function casts.
using GenericFunctionPointer = void (*)(void);

enum class PassBy { Ref, Val };
template<typename TA, PassBy Pass = PassBy::Ref> struct ArgumentInfo {
  using Type = TA;
  static constexpr PassBy pass{Pass};
};

template<typename TR, typename... ArgInfo> struct Signature {
  // Note valid template argument are of form
  //<TR, ArgumentInfo<TA, PassBy>...> where TA and TR belong to RuntimeTypes.
  // RuntimeTypes is a type union defined in intrinsics-library-templates.h to
  // avoid circular dependencies. Argument of type void cannot be passed by
  // value. So far TR cannot be a pointer.
  const std::string name;
};

struct IntrinsicProcedureRuntimeDescription {
  const std::string name;
  const TypeCode returnType;
  const std::vector<TypeCode> argumentsType;
  const std::vector<PassBy> argumentsPassedBy;
  const bool isElemental;
  const GenericFunctionPointer callable;
  // Construct from description using host independent types (RuntimeTypes)
  template<typename TR, typename... ArgInfo>
  IntrinsicProcedureRuntimeDescription(
      const Signature<TR, ArgInfo...> &signature, bool isElemental = false);
};

// HostRuntimeIntrinsicProcedure allows host runtime function to be called for
// constant folding.
struct HostRuntimeIntrinsicProcedure : IntrinsicProcedureRuntimeDescription {
  // Construct from runtime pointer with host types (float, double....)
  template<typename HostTR, typename... HostTA>
  HostRuntimeIntrinsicProcedure(const std::string &name,
      FuncPointer<HostTR, HostTA...> func, bool isElemental = false);
  HostRuntimeIntrinsicProcedure(
      const IntrinsicProcedureRuntimeDescription &rteProc,
      GenericFunctionPointer handle)
    : IntrinsicProcedureRuntimeDescription{rteProc}, handle{handle} {}
  GenericFunctionPointer handle;
};

// Defines a wrapper type that indirects calls to host runtime functions.
// Valid ConstantContainer are Scalar (only for elementals) and Constant.
template<template<typename> typename ConstantContainer, typename TR,
    typename... TA>
using HostProcedureWrapper = std::function<ConstantContainer<TR>(
    FoldingContext &, ConstantContainer<TA>...)>;

// HostIntrinsicProceduresLibrary is a data structure that holds
// HostRuntimeIntrinsicProcedure elements. It is meant for constant folding.
// When queried for an intrinsic procedure, it can return a callable object that
// implements this intrinsic if a host runtime function pointer for this
// intrinsic was added to its data structure.
class HostIntrinsicProceduresLibrary {
public:
  HostIntrinsicProceduresLibrary();
  void AddProcedure(HostRuntimeIntrinsicProcedure &&sym) {
    const std::string name{sym.name};
    procedures_.insert(std::make_pair(name, std::move(sym)));
  }
  bool HasEquivalentProcedure(
      const IntrinsicProcedureRuntimeDescription &sym) const;
  template<template<typename> typename ConstantContainer, typename TR,
      typename... TA>
  std::optional<HostProcedureWrapper<ConstantContainer, TR, TA...>>
  GetHostProcedureWrapper(const std::string &name) const;

private:
  std::multimap<std::string, const HostRuntimeIntrinsicProcedure> procedures_;
};

}
#endif  // FORTRAN_EVALUATE_INTRINSICS_LIBRARY_H_
