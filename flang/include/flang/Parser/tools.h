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

// When a parse tree node is an instance of a specific type wrapped in
// layers of packaging, return a pointer to that object.
// Implemented with mutually recursive template functions that are
// wrapped in a struct to avoid prototypes.
struct UnwrapperHelper {

  template<typename A, typename B> static const A *Unwrap(B *p) {
    if (p) {
      return Unwrap<A>(*p);
    } else {
      return nullptr;
    }
  }

  template<typename A, typename B, bool COPY>
  static const A *Unwrap(const common::Indirection<B, COPY> &x) {
    return Unwrap<A>(x.value());
  }

  template<typename A, typename... Bs>
  static const A *Unwrap(const std::variant<Bs...> &x) {
    return std::visit([](const auto &y) { return Unwrap<A>(y); }, x);
  }

  template<typename A, typename B>
  static const A *Unwrap(const std::optional<B> &o) {
    if (o) {
      return Unwrap<A>(*o);
    } else {
      return nullptr;
    }
  }

  template<typename A, typename B> static const A *Unwrap(B &x) {
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

template<typename A, typename B> const A *Unwrap(const B &x) {
  return UnwrapperHelper::Unwrap<A>(x);
}

// Get the CoindexedNamedObject if the entity is a coindexed object.
const CoindexedNamedObject *GetCoindexedNamedObject(const AllocateObject &);
const CoindexedNamedObject *GetCoindexedNamedObject(const DataRef &);

}
#endif  // FORTRAN_PARSER_TOOLS_H_
