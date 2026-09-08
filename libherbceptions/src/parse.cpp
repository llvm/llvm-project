//===--- parse.cpp - parse_errc error domain ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the parse_errc error_domain_singleton vtable and the weak
// __cxa_error_domain_parse ABI entry point. Available on all platforms.
// Name/message strings come from src/parse_table.hpp through
// simple_query_information_common.h (which_errc slot 2).
//
//===----------------------------------------------------------------------===//

#include "simple_query_information_common.h"

namespace {
constinit ::std::error_domain_singleton parse_error_domain{
    .do_equivalent =
        [](::std::size_t cd, ::std::error_domain_singleton const *otherdomain,
           ::std::size_t othercd) noexcept {
          if (otherdomain == __builtin_addressof(parse_error_domain))
            return cd == othercd;
          return parse_error_domain.do_to_errc(cd) ==
                 otherdomain->do_to_errc(othercd);
        },
    .do_query_information =
        [](::std::size_t cd, ::std::error_query_information query,
           ::std::error_reporter_encoding encoding, void *cookie,
           ::std::error_reporter_io_cookie_function cookfun) noexcept {
          ::std::error_domains::__herbceptions_detail::
              __simple_query_information_common(cd, query, encoding, cookie,
                                                cookfun, 2);
        },
    .do_to_errc = [](::std::size_t cd) noexcept -> ::std::errc {
      switch (static_cast<::std::parse_errc>(cd)) {
      case ::std::parse_errc::invalid:
        return ::std::errc::invalid_argument;
      case ::std::parse_errc::overflow:
        return ::std::errc::result_out_of_range;
      case ::std::parse_errc::partial:
        return ::std::errc::resource_unavailable_try_again;
      case ::std::parse_errc::ok:
        return ::std::errc{}; // ERRNO_SUCCESS must be zero or syscall breaks
      case ::std::parse_errc::end_of_file:
        break;
      }
      return ::std::errc::io_error;
    }};
} // namespace

extern "C" __HERBCEPTIONS_API ::std::error_domain_singleton const *
__cxa_error_domain_parse() noexcept {
  return __builtin_addressof(parse_error_domain);
}
