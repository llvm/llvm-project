//===--- win32_domain.cpp - win32 (win32_errc) error domain ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the win32 (win32_errc, Win32 GetLastError codes) error_domain
// singleton vtable and the weak __cxa_error_domain_win32 ABI entry point.
// Only built on _WIN32/__CYGWIN__ targets.
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32) || defined(__CYGWIN__)

#include "libherbceptions.h"
#include "ntkernel.h"
#include "win32_imports.h"
#include "win32_message_text.h"

namespace {

inline constexpr ::std::io_scatter_t
win32_name_message_range(::std::error_reporter_encoding encoding,
                         ::std::size_t startpos, ::std::size_t n) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    return {&startpos["\xAD\x97\x96\xA2\x89\xA7\xBD"], n};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    return {&startpos[u"[win32]"], n * sizeof(char16_t)};
  }
  case ::std::error_reporter_encoding::utf32: {
    return {&startpos[U"[win32]"], n * sizeof(char32_t)};
  }
  default: {
    return {&startpos[u8"[win32]"], n};
  }
  }
}

inline constexpr ::std::io_scatter_t
win32_name(::std::error_reporter_encoding encoding) noexcept {
  return win32_name_message_range(encoding, 1u, 5u);
}

inline constexpr ::std::io_scatter_t
win32_name_message(::std::error_reporter_encoding encoding) noexcept {
  return win32_name_message_range(encoding, 0u, 7u);
}

/*
Writes "(0x<hex code>)" for the requested encoding into __numbuf and returns
it as a scatter. The buffer must be at least
__format_hex_value_max_size_with_brackets<::std::uint_least32_t> code units
wide, each of the largest supported character size.
*/
inline constexpr ::std::io_scatter_t
win32_code_scatter(::std::error_reporter_encoding encoding,
                   ::std::uint_least32_t win32err,
                   char unsigned *__numbuf) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    auto *__dest{::std::error_domains::__herbceptions_detail::
                     __format_hex_value_full_with_bracket<true, char unsigned>(
                         __numbuf, win32err)};
    return {__numbuf, static_cast<::std::size_t>(__dest - __numbuf)};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    using __char16_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
        [[__gnu__::__may_alias__]]
#endif
        = char16_t *;
    auto *__dest{
        ::std::error_domains::__herbceptions_detail::
            __format_hex_value_full_with_bracket<false, char16_t>(
                reinterpret_cast<__char16_may_alias_ptr>(__numbuf), win32err)};
    return {__numbuf,
            static_cast<::std::size_t>(
                reinterpret_cast<char unsigned *>(__dest) - __numbuf)};
  }
  case ::std::error_reporter_encoding::utf32: {
    using __char32_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
        [[__gnu__::__may_alias__]]
#endif
        = char32_t *;
    auto *__dest{
        ::std::error_domains::__herbceptions_detail::
            __format_hex_value_full_with_bracket<false, char32_t>(
                reinterpret_cast<__char32_may_alias_ptr>(__numbuf), win32err)};
    return {__numbuf,
            static_cast<::std::size_t>(
                reinterpret_cast<char unsigned *>(__dest) - __numbuf)};
  }
  default: {
    auto *__dest{::std::error_domains::__herbceptions_detail::
                     __format_hex_value_full_with_bracket<false, char unsigned>(
                         __numbuf, win32err)};
    return {__numbuf, static_cast<::std::size_t>(__dest - __numbuf)};
  }
  }
}

constinit ::std::error_domain_singleton win32_error_domain{
    .do_equivalent =
        [](::std::size_t cd, ::std::error_domain_singleton const *otherdomain,
           ::std::size_t othercd) noexcept {
          using namespace ::std::error_domains::__herbceptions_detail;
          // win32 <-> win32: identity.
          if (otherdomain == __builtin_addressof(win32_error_domain))
            return cd == othercd;
          auto const win32err{static_cast<::std::uint_least32_t>(cd)};
          // win32 <-> nt: exact match on the table's win32 column.
          if (otherdomain == ::std::error_domains::__cxa_error_domain_nt())
            return __nt_win32_equivalent(
                       win32err, static_cast<::std::uint_least32_t>(othercd)) ==
                   1;
          // win32 <-> com: only a FACILITY_WIN32 HRESULT without the NT bit
          // equates to its embedded Win32 code; other combinations compare
          // through std::errc below.
          if (otherdomain == ::std::error_domains::__cxa_error_domain_com()) {
            auto const rule{__com_win32_equivalent(
                static_cast<::std::uint_least32_t>(othercd), win32err)};
            if (rule >= 0)
              return rule == 1;
          }
          return win32_error_domain.do_to_errc(cd) ==
                 otherdomain->do_to_errc(othercd);
        },
    .do_query_information =
        [](::std::size_t cd, ::std::error_query_information query,
           ::std::error_reporter_encoding encoding, void *cookie,
           ::std::error_reporter_io_cookie_function cookfun) noexcept {
          if (static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::name_message) <
              static_cast<::std::uint_least32_t>(query)) {
            return;
          }
          ::std::uint_least32_t win32err{
              static_cast<::std::uint_least32_t>(cd)};
          if constexpr (::std::error_domains::__herbceptions_detail::
                            __is_freestanding_kernel_mode) {
            /*
            Freestanding kernel mode: FormatMessage and its message tables
            are unavailable / a waste of rom size, so only
            [win32](0x<hex code>) is ever reported. Only a small fixed
            scratch buffer is used.
            */
            ::std::io_scatter_t scatters[2];
            ::std::size_t scatterlen{};
            switch (query) {
            case ::std::error_query_information::name: {
              *scatters = win32_name(encoding);
              scatterlen = 1u;
              break;
            }
            case ::std::error_query_information::message:
              [[fallthrough]];
            case ::std::error_query_information::name_message: {
              alignas(char32_t) char unsigned
                  __numbuf[::std::error_domains::__herbceptions_detail::
                               __format_hex_value_max_size_with_brackets<
                                   ::std::uint_least32_t> *
                           sizeof(char32_t)];
              scatterlen = 0u;
              if (::std::error_query_information::name_message == query) {
                *scatters = win32_name_message(encoding);
                ++scatterlen;
              }
              scatters[scatterlen] =
                  win32_code_scatter(encoding, win32err, __numbuf);
              ++scatterlen;
              break;
            }
            }
            cookfun(cookie, scatters, scatterlen);
          } else {
            ::std::io_scatter_t scatters[2];
            ::std::size_t scatterlen{};
            switch (query) {
            case ::std::error_query_information::name: {
              *scatters = win32_name(encoding);
              scatterlen = 1u;
              break;
            }
            case ::std::error_query_information::message:
              [[fallthrough]];
            case ::std::error_query_information::name_message: {
              alignas(char32_t) char unsigned
                  __numbuf[::std::error_domains::__herbceptions_detail::
                               __format_hex_value_max_size_with_brackets<
                                   ::std::uint_least32_t> *
                           sizeof(char32_t)];
              scatterlen = 0u;
              if (::std::error_query_information::name_message == query) {
                *scatters = win32_name_message(encoding);
                ++scatterlen;
              }
              scatters[scatterlen] =
                  win32_code_scatter(encoding, win32err, __numbuf);
              ++scatterlen;
              cookfun(cookie, scatters, scatterlen);
              ::std::error_domains::__herbceptions_detail::
                  __report_win32_message_text(win32err, encoding, cookie,
                                              cookfun);
              return;
            }
            }
            cookfun(cookie, scatters, scatterlen);
          }
        },
    .do_to_errc = [](::std::size_t cd) noexcept -> ::std::errc {
      return ::std::error_domains::__herbceptions_detail::__win32_to_errc(
          static_cast<::std::uint_least32_t>(cd));
    }};
} // namespace

extern "C" __HERBCEPTIONS_API ::std::error_domain_singleton const *
__cxa_error_domain_win32() noexcept {
  return __builtin_addressof(win32_error_domain);
}

#endif // _WIN32 || __CYGWIN__
