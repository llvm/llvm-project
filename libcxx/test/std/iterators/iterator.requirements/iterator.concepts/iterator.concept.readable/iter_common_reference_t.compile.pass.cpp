//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// iter_common_reference_t

#include <iterator>

#include <concepts>

struct X { };

// value_type and dereferencing are the same
struct T1 {
  using value_type = X;
  X operator*() const;
};
static_assert(std::same_as<std::iter_common_reference_t<T1>, X>);

// value_type and dereferencing are the same (modulo qualifiers)
struct T2 {
  using value_type = X;
  X& operator*() const;
};
static_assert(std::same_as<std::iter_common_reference_t<T2>, X&>);

// There's a custom common reference between value_type and the type of dereferencing
struct A { };
struct B { };
struct Common { Common(A); Common(B); };
template <template <class> class TQual, template <class> class QQual>
struct std::basic_common_reference<A, B, TQual, QQual> {
  using type = Common;
};
template <template <class> class TQual, template <class> class QQual>
struct std::basic_common_reference<B, A, TQual, QQual>
  : std::basic_common_reference<A, B, TQual, QQual>
{ };

struct T3 {
  using value_type = A;
  B&& operator*() const;
};
static_assert(std::same_as<std::iter_common_reference_t<T3>, Common>);

// Make sure we're SFINAE-friendly
template <class T>
constexpr bool has_common_reference = requires {
  typename std::iter_common_reference_t<T>;
};
struct NotIndirectlyReadable { };
static_assert(!has_common_reference<NotIndirectlyReadable>);
