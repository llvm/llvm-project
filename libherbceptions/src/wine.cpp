//===--- wine.cpp - wine (wine_errc) error domain
//--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the wine (wine_errc) error_domain_singleton vtable and the weak
// __cxa_error_domain_wine ABI entry point. Only built on _WIN32/__CYGWIN__
// targets. wine_errc uses Wine's own UNIX errno values (Linux-kernel style
// numbering); name/message strings come from src/wine_table.hpp through
// simple_query_information_common.h (which_errc slot 3).
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32) || defined(__CYGWIN__)

#include "simple_query_information_common.h"

namespace {
constinit ::std::error_domain_singleton __wine_error_domain{
    .do_cleanup = nullptr,
    .do_equivalent =
        [](::std::size_t cd, ::std::error_domain_singleton const *otherdomain,
           ::std::size_t othercd) noexcept {
          return __wine_error_domain.do_to_errc(cd) ==
                 otherdomain->do_to_errc(othercd);
        },
    .do_query_information =
        [](::std::size_t cd, ::std::error_query_information query,
           ::std::error_reporter_encoding encoding, void *cookie,
           ::std::error_reporter_io_cookie_function cookfun) noexcept {
          ::std::error_domains::__herbceptions_detail::
              __simple_query_information_common(cd, query, encoding, cookie,
                                                cookfun, 3);
        },
    // Wine's UNIX errno values only partially overlap std::errc, and some
    // wine.h names disagree with their errno semantics. The generated map
    // (guarded on the cerrno macros) switches on the wine_errc enumerator
    // and returns the semantically matching errc; anything without an errc
    // meaning falls back to io_error.
    .do_to_errc = [](::std::size_t cd) noexcept -> ::std::errc {
      switch (static_cast<::std::uint_least32_t>(cd)) {
#include "wine_errc_map.hpp"
      }
      return ::std::errc::io_error;
    }};
} // namespace

extern "C" __HERBCEPTIONS_API ::std::error_domain_singleton const *
__cxa_error_domain_wine() noexcept {
  return __builtin_addressof(__wine_error_domain);
}

#endif // _WIN32 || __CYGWIN__
