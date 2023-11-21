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

#include <array>
#include <string>

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
  [[maybe_unused]] static inline std::string_view EnumToString(NAME e) { \
    static const constexpr char vaArgs[]{#__VA_ARGS__}; \
    static const constexpr auto names{ \
        ::Fortran::common::EnumNames<NAME##_enumSize>(vaArgs)}; \
    return names[static_cast<std::size_t>(e)]; \
  }

} // namespace Fortran::common
#endif // FORTRAN_COMMON_ENUM_CLASS_H_
