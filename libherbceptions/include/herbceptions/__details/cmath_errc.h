//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#if __has_include(<math.h>)
#include <math.h>
#endif

namespace std {
enum class cmath_errc : ::std::uint_least32_t {
#ifdef _FE_INVALID
  invalid = _FE_INVALID,
  divbyzero = _FE_DIVBYZERO,
  inexact = _FE_INEXACT,
  overflow = _FE_OVERFLOW,
  underflow = _FE_UNDERFLOW,
#elif defined(FE_INVALID)
  invalid = FE_INVALID,
  divbyzero = FE_DIVBYZERO,
  inexact = FE_INEXACT,
  overflow = FE_OVERFLOW,
  underflow = FE_UNDERFLOW,
#else
  invalid = 1,
  divbyzero = 2,
  inexact = 4,
  overflow = 8,
  underflow = 16,
#endif
  all_except = divbyzero | inexact | invalid | overflow | underflow
};

inline constexpr bool operator==(::std::cmath_errc __a,
                                 ::std::cmath_errc __b) noexcept {
  return static_cast<::std::uint_least32_t>(__a) ==
         static_cast<::std::uint_least32_t>(__b);
}

template <> class error_domain<::std::cmath_errc> {
public:
  using errc_type = ::std::cmath_errc;
  static inline constexpr ::std::error_domain_singleton const *
  domain() noexcept {
    return ::std::error_domains::__cxa_error_domain_parse();
  }
  static inline constexpr ::std::size_t code(errc_type __e) noexcept {
    return static_cast<::std::size_t>(static_cast<::std::uint_least32_t>(__e));
  }
};
} // namespace std
