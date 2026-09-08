//===--- error_codes_fuzzer.cpp - error domain code fuzzer ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Drives every available error domain's do_query_information (all queries x
// all encodings), do_to_errc and pairwise do_equivalent with attacker
// controlled codes. Everything here is memory safe by construction; the
// point is to shake out formatting / conversion / table bugs.
//
//===----------------------------------------------------------------------===//

#include "fuzz_harness.h"

namespace {

using ::std::error_domain_singleton;
using ::std::error_query_information;
using ::std::error_reporter_encoding;

constexpr ::std::error_query_information kQueries[] = {
    error_query_information::name,
    error_query_information::message,
    error_query_information::name_message,
};

constexpr ::std::error_reporter_encoding kEncodings[] = {
    error_reporter_encoding::utf8,
    error_reporter_encoding::utf16,
    error_reporter_encoding::utf32,
    error_reporter_encoding::gb18030,
    error_reporter_encoding::utfebcdic,
};

void query_all(error_domain_singleton const *domain,
               ::std::uint_least32_t code) noexcept {
  herbceptions_fuzz::capture_ctx ctx;
  for (auto const q : kQueries) {
    for (auto const e : kEncodings) {
      herbceptions_fuzz::reset(ctx);
      domain->do_query_information(code, q, e, &ctx,
                                   herbceptions_fuzz::capture_cookfun);
    }
  }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(::std::uint8_t const *data,
                                      ::std::size_t size) {
  if (size < 8) {
    return 0;
  }
  auto const a{static_cast<::std::uint_least32_t>(
      data[0] | (data[1] << 8u) | (data[2] << 16u) | (data[3] << 24u))};
  auto const b{static_cast<::std::uint_least32_t>(
      data[4] | (data[5] << 8u) | (data[6] << 16u) | (data[7] << 24u))};

  query_all(::std::error_domains::__cxa_error_domain_posix(), a);
  query_all(::std::error_domains::__cxa_error_domain_parse(), b);
  query_all(::std::error_domains::__cxa_error_domain_cmath(), a ^ b);

#ifdef _WIN32
  auto const *wine{::std::error_domains::__cxa_error_domain_wine()};
  auto const *win32{::std::error_domains::__cxa_error_domain_win32()};
  auto const *nt{::std::error_domains::__cxa_error_domain_nt()};
  auto const *com{::std::error_domains::__cxa_error_domain_com()};

  constexpr ::std::uint_least32_t kComFacilityNtBit{0x10000000u};
  constexpr ::std::uint_least32_t kComFacilityWin32Base{0x80070000u};

  query_all(wine, b);
  query_all(win32, a);
  query_all(nt, a);
  query_all(nt, 0x80000000u | (a & 0x7FFFFFFFu));
  query_all(com, b);
  query_all(com, b | kComFacilityNtBit);
  query_all(com, kComFacilityWin32Base | (a & 0xFFFFu));

  // Cross-domain equivalence over the whole triangle plus posix/wine.
  error_domain_singleton const *const domains[] = {
      ::std::error_domains::__cxa_error_domain_posix(), wine, win32, nt, com};
  for (auto const *d1 : domains) {
    for (auto const *d2 : domains) {
      d1->do_equivalent(a, d2, b);
      d1->do_to_errc(a);
    }
  }
#endif
  return 0;
}
