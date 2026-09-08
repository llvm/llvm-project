//===--- posix.cpp - posix (std::errc) error domain -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the posix (std::errc) error_domain_singleton vtable and the weak
// __cxa_error_domain_posix ABI entry point. Available on all platforms.
// The errno message table lives in src/libherbceptions.h.
//
//===----------------------------------------------------------------------===//

#include "simple_query_information_common.h"

namespace {
constinit ::std::error_domain_singleton posix_error_domain{
    .do_equivalent =
        [](::std::size_t cd, ::std::error_domain_singleton const *otherdomain,
           ::std::size_t othercd) noexcept {
          return posix_error_domain.do_to_errc(cd) ==
                 otherdomain->do_to_errc(othercd);
        },
    .do_query_information =
        [](::std::size_t cd, ::std::error_query_information query,
           ::std::error_reporter_encoding encoding, void *cookie,
           ::std::error_reporter_io_cookie_function cookfun) noexcept {
          ::std::error_domains::__herbceptions_detail::
              __simple_query_information_common(cd, query, encoding, cookie,
                                                cookfun, 0);
        },
    .do_to_errc =
        [](::std::size_t cd) noexcept { return static_cast<::std::errc>(cd); }};
} // namespace

extern "C" __HERBCEPTIONS_API ::std::error_domain_singleton const *
__cxa_error_domain_posix() noexcept {
  return __builtin_addressof(posix_error_domain);
}
