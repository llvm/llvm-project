//===-- include/flang/Common/unwrap.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_UNWRAP_H_
#define FORTRAN_COMMON_UNWRAP_H_

#include "indirection.h"
#include "reference-counted.h"
#include "reference.h"
#include "variant.h"
#include "visit.h"
#include <memory>
#include <optional>
#include <type_traits>

// Given a nest of variants, optionals, &/or pointers, Unwrap<>() isolates
// a packaged value of a specific type if it is present and returns a pointer
// thereto; otherwise, it returns a null pointer.  It's analogous to
// std::get_if<>() but it accepts a reference argument and is recursive.
// The target type parameter cannot be omitted.
//
// Be advised: If the target type parameter is not const-qualified, but the
// isolated value is const-qualified, the result of Unwrap<> will be a
// pointer to a const-qualified value.
//
// Further: const-qualified alternatives in instances of non-const-qualified
// variants will not be returned from Unwrap if the target type is not
// const-qualified.
//
// UnwrapCopy<>() is a variation of Unwrap<>() that returns an optional copy
// of the value if one is present with the desired type.

namespace Fortran::common {

// Utility: Produces "const A" if B is const and A is not already so.
template <typename A, typename B>
using Constify = std::conditional_t<std::is_const_v<B> && !std::is_const_v<A>,
    std::add_const_t<A>, A>;

// Unwrap's mutually-recursive template functions are packaged in a struct
// to avoid a need for prototypes.
struct UnwrapperHelper {

  // Base case
  template <typename A, typename B>
  static auto Unwrap(B &x) -> Constify<A, B> * {
    if constexpr (std::is_same_v<std::decay_t<A>, std::decay_t<B>>) {
      return &x;
    } else {
      return nullptr;
    }
  }

  // Implementations of specializations
  template <typename A, typename B>
  static auto Unwrap(B *p) -> Constify<A, B> * {
    if (p) {
      return Unwrap<A>(*p);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename B>
  static auto Unwrap(const std::unique_ptr<B> &p) -> Constify<A, B> * {
    if (p.get()) {
      return Unwrap<A>(*p);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename B>
  static auto Unwrap(const std::shared_ptr<B> &p) -> Constify<A, B> * {
    if (p.get()) {
      return Unwrap<A>(*p);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename B>
  static auto Unwrap(std::optional<B> &x) -> Constify<A, B> * {
    if (x) {
      return Unwrap<A>(*x);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename B>
  static auto Unwrap(const std::optional<B> &x) -> Constify<A, B> * {
    if (x) {
      return Unwrap<A>(*x);
    } else {
      return nullptr;
    }
  }

  template <typename A, typename... Bs>
  static A *Unwrap(std::variant<Bs...> &u) {
    return common::visit(
        [](auto &x) -> A * {
          using Ty = std::decay_t<decltype(Unwrap<A>(x))>;
          if constexpr (!std::is_const_v<std::remove_pointer_t<Ty>> ||
              std::is_const_v<A>) {
            return Unwrap<A>(x);
          }
          return nullptr;
        },
        u);
  }

  template <typename A, typename... Bs>
  static auto Unwrap(const std::variant<Bs...> &u) -> std::add_const_t<A> * {
    return common::visit(
        [](const auto &x) -> std::add_const_t<A> * { return Unwrap<A>(x); }, u);
  }

  template <typename A, typename B>
  static auto Unwrap(const Reference<B> &ref) -> Constify<A, B> * {
    return Unwrap<A>(*ref);
  }

  template <typename A, typename B, bool COPY>
  static auto Unwrap(const Indirection<B, COPY> &p) -> Constify<A, B> * {
    return Unwrap<A>(p.value());
  }

  template <typename A, typename B>
  static auto Unwrap(const CountedReference<B> &p) -> Constify<A, B> * {
    if (p.get()) {
      return Unwrap<A>(*p);
    } else {
      return nullptr;
    }
  }
};

template <typename A, typename B> auto Unwrap(B &x) -> Constify<A, B> * {
  return UnwrapperHelper::Unwrap<A>(x);
}

// Returns a copy of a wrapped value, if present, otherwise a vacant optional.
template <typename A, typename B> std::optional<A> UnwrapCopy(const B &x) {
  if (const A * p{Unwrap<A>(x)}) {
    return std::make_optional<A>(*p);
  } else {
    return std::nullopt;
  }
}
} // namespace Fortran::common
#endif // FORTRAN_COMMON_UNWRAP_H_
