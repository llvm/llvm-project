//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_FORMAT_FORMAT_FORMAT_ARGUMENTS_FORMAT_ARG_VISITORS_H
#define TEST_STD_UTILITIES_FORMAT_FORMAT_FORMAT_ARGUMENTS_FORMAT_ARG_VISITORS_H

#include <concepts>
#include <format>
#include <string_view>
#include <variant>

// [format.arg]
// namespace std {
//   template<class Context>
//   class basic_format_arg {
//   public:
//     class handle;
//   private:
//     [...]
//     using char_type = typename Context::char_type;                              // exposition only
//
//     variant<monostate, bool, char_type,
//             int, unsigned int, long long int, unsigned long long int,
//             float, double, long double,
//             const char_type*, basic_string_view<char_type>,
//             const void*, handle> value;                                         // exposition only
//     [...]
//   };
// }

template <class T, class Context>
concept format_arg_visit_type_for =
    std::same_as<T, std::monostate> || std::same_as<T, bool> || std::same_as<T, typename Context::char_type> ||
    std::same_as<T, int> || std::same_as<T, unsigned int> || std::same_as<T, long long> ||
    std::same_as<T, unsigned long long> || std::same_as<T, float> || std::same_as<T, double> ||
    std::same_as<T, long double> || std::same_as<T, const typename Context::char_type*> ||
    std::same_as<T, std::basic_string_view<typename Context::char_type>> || std::same_as<T, const void*> ||
    std::same_as<T, typename std::basic_format_arg<Context>::handle>;

// Verify that visitors don't see other types.

template <class Context>
struct limited_visitor {
  void operator()(auto) const = delete;

  void operator()(format_arg_visit_type_for<Context> auto) const {}
};

template <class Context>
struct limited_int_visitor {
  void operator()(auto) const = delete;

  int operator()(format_arg_visit_type_for<Context> auto) const { return 42; }
};

#endif // TEST_STD_UTILITIES_FORMAT_FORMAT_FORMAT_ARGUMENTS_FORMAT_ARG_VISITORS_H
