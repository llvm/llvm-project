//===-- include/flang/Parser/tools.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_TOOLS_H_
#define FORTRAN_PARSER_TOOLS_H_

#include "parse-tree.h"

namespace Fortran::parser {

// GetLastName() isolates and returns a reference to the rightmost Name
// in a variable (i.e., the Name whose symbol's type determines the type
// of the variable or expression).
const Name &GetLastName(const Name &);
const Name &GetLastName(const StructureComponent &);
const Name &GetLastName(const DataRef &);
const Name &GetLastName(const Substring &);
const Name &GetLastName(const Designator &);
const Name &GetLastName(const ProcComponentRef &);
const Name &GetLastName(const ProcedureDesignator &);
const Name &GetLastName(const Call &);
const Name &GetLastName(const FunctionReference &);
const Name &GetLastName(const Variable &);
const Name &GetLastName(const AllocateObject &);

// GetFirstName() isolates and returns a reference to the leftmost Name
// in a variable or entity declaration.
const Name &GetFirstName(const Name &);
const Name &GetFirstName(const StructureComponent &);
const Name &GetFirstName(const DataRef &);
const Name &GetFirstName(const Substring &);
const Name &GetFirstName(const Designator &);
const Name &GetFirstName(const ProcComponentRef &);
const Name &GetFirstName(const ProcedureDesignator &);
const Name &GetFirstName(const Call &);
const Name &GetFirstName(const FunctionReference &);
const Name &GetFirstName(const Variable &);

// When a parse tree node is an instance of a specific type wrapped in
// layers of packaging, return a pointer to that object.
// Implemented with mutually recursive template functions that are
// wrapped in a struct to avoid prototypes.
struct UnwrapperHelper {

  template <typename A, typename B> static const A *Unwrap(B *p) {
    if (p) {
      return Unwrap<A>(*p);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename B, bool COPY>
  static const A *Unwrap(const common::Indirection<B, COPY> &x) {
    return Unwrap<A>(x.value());
  }

  template <typename A, typename... Bs>
  static const A *Unwrap(const std::variant<Bs...> &x) {
    return std::visit([](const auto &y) { return Unwrap<A>(y); }, x);
  }

  template <typename A, typename B>
  static const A *Unwrap(const std::optional<B> &o) {
    if (o) {
      return Unwrap<A>(*o);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename B>
  static const A *Unwrap(const UnlabeledStatement<B> &x) {
    return Unwrap<A>(x.statement);
  }
  template <typename A, typename B>
  static const A *Unwrap(const Statement<B> &x) {
    return Unwrap<A>(x.statement);
  }

  template <typename A, typename B> static const A *Unwrap(B &x) {
    if constexpr (std::is_same_v<std::decay_t<A>, std::decay_t<B>>) {
      return &x;
    } else if constexpr (ConstraintTrait<B>) {
      return Unwrap<A>(x.thing);
    } else if constexpr (WrapperTrait<B>) {
      return Unwrap<A>(x.v);
    } else if constexpr (UnionTrait<B>) {
      return Unwrap<A>(x.u);
    } else {
      return nullptr;
    }
  }
};

template <typename A, typename B> const A *Unwrap(const B &x) {
  return UnwrapperHelper::Unwrap<A>(x);
}
template <typename A, typename B> A *Unwrap(B &x) {
  return const_cast<A *>(Unwrap<A, B>(const_cast<const B &>(x)));
}

// Get the CoindexedNamedObject if the entity is a coindexed object.
const CoindexedNamedObject *GetCoindexedNamedObject(const AllocateObject &);
const CoindexedNamedObject *GetCoindexedNamedObject(const DataRef &);
const CoindexedNamedObject *GetCoindexedNamedObject(const Designator &);
const CoindexedNamedObject *GetCoindexedNamedObject(const Variable &);

// Detects parse tree nodes with "source" members.
template <typename A, typename = int> struct HasSource : std::false_type {};
template <typename A>
struct HasSource<A, decltype(static_cast<void>(A::source), 0)>
    : std::true_type {};

// Detects parse tree nodes with "typedExpr" members.
template <typename A, typename = int> struct HasTypedExpr : std::false_type {};
template <typename A>
struct HasTypedExpr<A, decltype(static_cast<void>(A::typedExpr), 0)>
    : std::true_type {};
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_TOOLS_H_
