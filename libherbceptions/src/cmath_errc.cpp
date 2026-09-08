//===--- cmath_errc.cpp - cmath_errc error domain
//--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the cmath_errc error_domain_singleton vtable and the weak
// __cxa_error_domain_cmath ABI entry point. Available on all platforms.
// Name/message strings come from src/cmath_table.hpp through
// simple_query_information_common.h (which_errc slot 1).
//
//===----------------------------------------------------------------------===//

#include "simple_query_information_common.h"

namespace {
constinit ::std::error_domain_singleton cmath_error_domain{
    .do_equivalent =
        [](::std::size_t cd, ::std::error_domain_singleton const *otherdomain,
           ::std::size_t othercd) noexcept {
          if (otherdomain == __builtin_addressof(cmath_error_domain))
            return cd == othercd;
          return cmath_error_domain.do_to_errc(cd) ==
                 otherdomain->do_to_errc(othercd);
        },
    .do_query_information =
        [](::std::size_t cd, ::std::error_query_information query,
           ::std::error_reporter_encoding encoding, void *cookie,
           ::std::error_reporter_io_cookie_function cookfun) noexcept {
          ::std::error_domains::__herbceptions_detail::
              __simple_query_information_common(cd, query, encoding, cookie,
                                                cookfun, 1);
        },
    .do_to_errc = [](::std::size_t cd) noexcept -> ::std::errc {
      switch (static_cast<::std::uint_least32_t>(cd)) {
      case static_cast<::std::uint_least32_t>(::std::cmath_errc::invalid):
        return ::std::errc::argument_out_of_domain;
      case static_cast<::std::uint_least32_t>(::std::cmath_errc::divbyzero):
      case static_cast<::std::uint_least32_t>(::std::cmath_errc::overflow):
      case static_cast<::std::uint_least32_t>(::std::cmath_errc::underflow):
        return ::std::errc::result_out_of_range;
      case static_cast<::std::uint_least32_t>(::std::cmath_errc::inexact):
        return ::std::errc{};
      default:
        return ::std::errc::io_error;
      }
    }};
} // namespace

extern "C" __HERBCEPTIONS_API ::std::error_domain_singleton const *
__cxa_error_domain_cmath() noexcept {
  return __builtin_addressof(cmath_error_domain);
}
