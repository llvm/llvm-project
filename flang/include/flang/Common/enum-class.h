//===-- include/flang/Common/enum-class.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The macro
//   ENUM_CLASS(className, enum1, enum2, ..., enumN)
// defines
//   enum class className { enum1, enum2, ... , enumN };
// as well as the introspective utilities
//   static constexpr std::size_t className_enumSize{N};
//   static inline std::string_view EnumToString(className);

#ifndef FORTRAN_COMMON_ENUM_CLASS_H_
#define FORTRAN_COMMON_ENUM_CLASS_H_

#include "optional.h"
#include <array>
#include <functional>
#include <string_view>
namespace Fortran::common {

constexpr std::size_t CountEnumNames(const char *p) {
  std::size_t n{0};
  std::size_t any{0};
  for (; *p; ++p) {
    if (*p == ',') {
      n += any;
      any = 0;
    } else if (*p != ' ') {
      any = 1;
    }
  }
  return n + any;
}

template <std::size_t ITEMS>
constexpr std::array<std::string_view, ITEMS> EnumNames(const char *p) {
  std::array<std::string_view, ITEMS> result{""};
  std::size_t at{0};
  const char *start{nullptr};
  for (; *p; ++p) {
    if (*p == ',' || *p == ' ') {
      if (start) {
        result[at++] =
            std::string_view{start, static_cast<std::size_t>(p - start)};
        start = nullptr;
      }
    } else if (!start) {
      start = p;
    }
  }
  if (start) {
    result[at] = std::string_view{start, static_cast<std::size_t>(p - start)};
  }
  return result;
}

#define ENUM_CLASS(NAME, ...) \
  enum class NAME { __VA_ARGS__ }; \
  [[maybe_unused]] static constexpr std::size_t NAME##_enumSize{ \
      ::Fortran::common::CountEnumNames(#__VA_ARGS__)}; \
  [[maybe_unused]] static constexpr std::array<std::string_view, \
      NAME##_enumSize> NAME##_names{ \
      ::Fortran::common::EnumNames<NAME##_enumSize>(#__VA_ARGS__)}; \
  [[maybe_unused]] static inline std::string_view EnumToString(NAME e) { \
    return NAME##_names[static_cast<std::size_t>(e)]; \
  }

namespace EnumClass {

using Predicate = std::function<bool(const std::string_view)>;
// Finds the first index for which the predicate returns true.
optional<std::size_t> FindIndex(
    Predicate pred, std::size_t size, const std::string_view *names);

using FindIndexType = std::function<optional<std::size_t>(Predicate)>;

template <typename NAME>
optional<NAME> inline Find(Predicate pred, FindIndexType findIndex) {
  return MapOption<int, NAME>(
      findIndex(pred), [](int x) { return static_cast<NAME>(x); });
}

} // namespace EnumClass

#define ENUM_CLASS_EXTRA(NAME) \
  [[maybe_unused]] inline optional<std::size_t> Find##NAME##Index( \
      ::Fortran::common::EnumClass::Predicate p) { \
    return ::Fortran::common::EnumClass::FindIndex( \
        p, NAME##_enumSize, NAME##_names.data()); \
  } \
  [[maybe_unused]] inline optional<NAME> Find##NAME( \
      ::Fortran::common::EnumClass::Predicate p) { \
    return ::Fortran::common::EnumClass::Find<NAME>(p, Find##NAME##Index); \
  } \
  [[maybe_unused]] inline optional<NAME> StringTo##NAME( \
      const std::string_view name) { \
    return Find##NAME( \
        [name](const std::string_view s) -> bool { return name == s; }); \
  }
} // namespace Fortran::common
#endif // FORTRAN_COMMON_ENUM_CLASS_H_
