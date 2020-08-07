//===-- include/flang/Common/template.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_COMMON_TEMPLATE_H_
#define FORTRAN_COMMON_TEMPLATE_H_

#include "flang/Common/idioms.h"
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

// Utility templates for metaprogramming and for composing the
// std::optional<>, std::tuple<>, and std::variant<> containers.

namespace Fortran::common {

// SearchTypeList<PREDICATE, TYPES...> scans a list of types.  The zero-based
// index of the first type T in the list for which PREDICATE<T>::value() is
// true is returned, or -1 if the predicate is false for every type in the list.
// This is a compile-time operation; see SearchTypes below for a run-time form.
template <int N, template <typename> class PREDICATE, typename TUPLE>
struct SearchTypeListHelper {
  static constexpr int value() {
    if constexpr (N >= std::tuple_size_v<TUPLE>) {
      return -1;
    } else if constexpr (PREDICATE<std::tuple_element_t<N, TUPLE>>::value()) {
      return N;
    } else {
      return SearchTypeListHelper<N + 1, PREDICATE, TUPLE>::value();
    }
  }
};

template <template <typename> class PREDICATE, typename... TYPES>
constexpr int SearchTypeList{
    SearchTypeListHelper<0, PREDICATE, std::tuple<TYPES...>>::value()};

// TypeIndex<A, TYPES...> scans a list of types for simple type equality.
// The zero-based index of A in the list is returned, or -1 if A is not present.
template <typename A> struct MatchType {
  template <typename B> struct Match {
    static constexpr bool value() {
      return std::is_same_v<std::decay_t<A>, std::decay_t<B>>;
    }
  };
};

template <typename A, typename... TYPES>
constexpr int TypeIndex{SearchTypeList<MatchType<A>::template Match, TYPES...>};

// IsTypeInList<A, TYPES...> is a simple presence predicate.
template <typename A, typename... TYPES>
constexpr bool IsTypeInList{TypeIndex<A, TYPES...> >= 0};

// OverMembers extracts the list of types that constitute the alternatives
// of a std::variant or elements of a std::tuple and passes that list as
// parameter types to a given variadic template.
template <template <typename...> class, typename> struct OverMembersHelper;
template <template <typename...> class T, typename... Ts>
struct OverMembersHelper<T, std::variant<Ts...>> {
  using type = T<Ts...>;
};
template <template <typename...> class T, typename... Ts>
struct OverMembersHelper<T, std::tuple<Ts...>> {
  using type = T<Ts...>;
};

template <template <typename...> class T, typename TUPLEorVARIANT>
using OverMembers =
    typename OverMembersHelper<T, std::decay_t<TUPLEorVARIANT>>::type;

// SearchMembers<PREDICATE> scans the types that constitute the alternatives
// of a std::variant instantiation or elements of a std::tuple.
// The zero-based index of the first type T among the alternatives for which
// PREDICATE<T>::value() is true is returned, or -1 when the predicate is false
// for every type in the set.
template <template <typename> class PREDICATE> struct SearchMembersHelper {
  template <typename... Ts> struct Scanner {
    static constexpr int value() { return SearchTypeList<PREDICATE, Ts...>; }
  };
};

template <template <typename> class PREDICATE, typename TUPLEorVARIANT>
constexpr int SearchMembers{
    OverMembers<SearchMembersHelper<PREDICATE>::template Scanner,
        TUPLEorVARIANT>::value()};

template <typename A, typename TUPLEorVARIANT>
constexpr bool HasMember{
    SearchMembers<MatchType<A>::template Match, TUPLEorVARIANT> >= 0};

// std::optional<std::optional<A>> -> std::optional<A>
template <typename A>
std::optional<A> JoinOptional(std::optional<std::optional<A>> &&x) {
  if (x) {
    return std::move(*x);
  }
  return std::nullopt;
}

// Convert an std::optional to an ordinary pointer
template <typename A> const A *GetPtrFromOptional(const std::optional<A> &x) {
  if (x) {
    return &*x;
  } else {
    return nullptr;
  }
}

// Copy a value from one variant type to another.  The types allowed in the
// source variant must all be allowed in the destination variant type.
template <typename TOV, typename FROMV> TOV CopyVariant(const FROMV &u) {
  return std::visit([](const auto &x) -> TOV { return {x}; }, u);
}

// Move a value from one variant type to another.  The types allowed in the
// source variant must all be allowed in the destination variant type.
template <typename TOV, typename FROMV>
common::IfNoLvalue<TOV, FROMV> MoveVariant(FROMV &&u) {
  return std::visit(
      [](auto &&x) -> TOV { return {std::move(x)}; }, std::move(u));
}

// CombineTuples takes a list of std::tuple<> template instantiation types
// and constructs a new std::tuple type that concatenates all of their member
// types.  E.g.,
//   CombineTuples<std::tuple<char, int>, std::tuple<float, double>>
// is std::tuple<char, int, float, double>.
template <typename... TUPLES> struct CombineTuplesHelper {
  static decltype(auto) f(TUPLES *...a) {
    return std::tuple_cat(std::move(*a)...);
  }
  using type = decltype(f(static_cast<TUPLES *>(nullptr)...));
};
template <typename... TUPLES>
using CombineTuples = typename CombineTuplesHelper<TUPLES...>::type;

// CombineVariants takes a list of std::variant<> instantiations and constructs
// a new instantiation that holds all of their alternatives, which must be
// pairwise distinct.
template <typename> struct VariantToTupleHelper;
template <typename... Ts> struct VariantToTupleHelper<std::variant<Ts...>> {
  using type = std::tuple<Ts...>;
};
template <typename VARIANT>
using VariantToTuple = typename VariantToTupleHelper<VARIANT>::type;

template <typename A, typename... REST> struct AreTypesDistinctHelper {
  static constexpr bool value() {
    if constexpr (sizeof...(REST) > 0) {
      // extra () for clang-format
      return ((... && !std::is_same_v<A, REST>)) &&
          AreTypesDistinctHelper<REST...>::value();
    }
    return true;
  }
};
template <typename... Ts>
constexpr bool AreTypesDistinct{AreTypesDistinctHelper<Ts...>::value()};

template <typename A, typename... Ts> struct AreSameTypeHelper {
  using type = A;
  static constexpr bool value() {
    if constexpr (sizeof...(Ts) == 0) {
      return true;
    } else {
      using Rest = AreSameTypeHelper<Ts...>;
      return std::is_same_v<type, typename Rest::type> && Rest::value();
    }
  }
};

template <typename... Ts>
constexpr bool AreSameType{AreSameTypeHelper<Ts...>::value()};

template <typename> struct TupleToVariantHelper;
template <typename... Ts> struct TupleToVariantHelper<std::tuple<Ts...>> {
  static_assert(AreTypesDistinct<Ts...>,
      "TupleToVariant: types are not pairwise distinct");
  using type = std::variant<Ts...>;
};
template <typename TUPLE>
using TupleToVariant = typename TupleToVariantHelper<TUPLE>::type;

template <typename... VARIANTS> struct CombineVariantsHelper {
  using type = TupleToVariant<CombineTuples<VariantToTuple<VARIANTS>...>>;
};
template <typename... VARIANTS>
using CombineVariants = typename CombineVariantsHelper<VARIANTS...>::type;

// SquashVariantOfVariants: given a std::variant whose alternatives are
// all std::variant instantiations, form a new union over their alternatives.
template <typename VARIANT>
using SquashVariantOfVariants = OverMembers<CombineVariants, VARIANT>;

// Given a type function, MapTemplate applies it to each of the types
// in a tuple or variant, and collect the results in a given variadic
// template (typically a std::variant).
template <template <typename> class, template <typename...> class, typename...>
struct MapTemplateHelper;
template <template <typename> class F, template <typename...> class PACKAGE,
    typename... Ts>
struct MapTemplateHelper<F, PACKAGE, std::tuple<Ts...>> {
  using type = PACKAGE<F<Ts>...>;
};
template <template <typename> class F, template <typename...> class PACKAGE,
    typename... Ts>
struct MapTemplateHelper<F, PACKAGE, std::variant<Ts...>> {
  using type = PACKAGE<F<Ts>...>;
};
template <template <typename> class F, typename TUPLEorVARIANT,
    template <typename...> class PACKAGE = std::variant>
using MapTemplate =
    typename MapTemplateHelper<F, PACKAGE, TUPLEorVARIANT>::type;

// std::tuple<std::optional<>...> -> std::optional<std::tuple<...>>
// i.e., inverts a tuple of optional values into an optional tuple that has
// a value only if all of the original elements were present.
template <typename... A, std::size_t... J>
std::optional<std::tuple<A...>> AllElementsPresentHelper(
    std::tuple<std::optional<A>...> &&t, std::index_sequence<J...>) {
  bool present[]{std::get<J>(t).has_value()...};
  for (std::size_t j{0}; j < sizeof...(J); ++j) {
    if (!present[j]) {
      return std::nullopt;
    }
  }
  return {std::make_tuple(*std::get<J>(t)...)};
}

template <typename... A>
std::optional<std::tuple<A...>> AllElementsPresent(
    std::tuple<std::optional<A>...> &&t) {
  return AllElementsPresentHelper(
      std::move(t), std::index_sequence_for<A...>{});
}

// std::vector<std::optional<A>> -> std::optional<std::vector<A>>
// i.e., inverts a vector of optional values into an optional vector that
// will have a value only when all of the original elements are present.
template <typename A>
std::optional<std::vector<A>> AllElementsPresent(
    std::vector<std::optional<A>> &&v) {
  for (const auto &maybeA : v) {
    if (!maybeA) {
      return std::nullopt;
    }
  }
  std::vector<A> result;
  for (auto &&maybeA : std::move(v)) {
    result.emplace_back(std::move(*maybeA));
  }
  return result;
}

// (std::optional<>...) -> std::optional<std::tuple<...>>
// i.e., given some number of optional values, return a optional tuple of
// those values that is present only of all of the values were so.
template <typename... A>
std::optional<std::tuple<A...>> AllPresent(std::optional<A> &&...x) {
  return AllElementsPresent(std::make_tuple(std::move(x)...));
}

// (f(A...) -> R) -> std::optional<A>... -> std::optional<R>
// Apply a function to optional arguments if all are present.
// N.B. If the function returns std::optional, MapOptional will return
// std::optional<std::optional<...>> and you will probably want to
// run it through JoinOptional to "squash" it.
template <typename R, typename... A>
std::optional<R> MapOptional(
    std::function<R(A &&...)> &&f, std::optional<A> &&...x) {
  if (auto args{AllPresent(std::move(x)...)}) {
    return std::make_optional(std::apply(std::move(f), std::move(*args)));
  }
  return std::nullopt;
}
template <typename R, typename... A>
std::optional<R> MapOptional(R (*f)(A &&...), std::optional<A> &&...x) {
  return MapOptional(std::function<R(A && ...)>{f}, std::move(x)...);
}

// Given a VISITOR class of the general form
//   struct VISITOR {
//     using Result = ...;
//     using Types = std::tuple<...>;
//     template<typename T> Result Test() { ... }
//   };
// SearchTypes will traverse the element types in the tuple in order
// and invoke VISITOR::Test<T>() on each until it returns a value that
// casts to true.  If no invocation of Test succeeds, SearchTypes will
// return a default value.
template <std::size_t J, typename VISITOR>
common::IfNoLvalue<typename VISITOR::Result, VISITOR> SearchTypesHelper(
    VISITOR &&visitor, typename VISITOR::Result &&defaultResult) {
  using Tuple = typename VISITOR::Types;
  if constexpr (J < std::tuple_size_v<Tuple>) {
    if (auto result{visitor.template Test<std::tuple_element_t<J, Tuple>>()}) {
      return result;
    }
    return SearchTypesHelper<J + 1, VISITOR>(
        std::move(visitor), std::move(defaultResult));
  } else {
    return std::move(defaultResult);
  }
}

template <typename VISITOR>
common::IfNoLvalue<typename VISITOR::Result, VISITOR> SearchTypes(
    VISITOR &&visitor,
    typename VISITOR::Result defaultResult = typename VISITOR::Result{}) {
  return SearchTypesHelper<0, VISITOR>(
      std::move(visitor), std::move(defaultResult));
}
} // namespace Fortran::common
#endif // FORTRAN_COMMON_TEMPLATE_H_
