//===--- com_domain.cpp - com (com_errc) error domain ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the com (com_errc, HRESULT codes) error_domain singleton vtable
// and the weak __cxa_error_domain_com ABI entry point. Only built on
// _WIN32/__CYGWIN__ targets.
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32) || defined(__CYGWIN__)

#include "libherbceptions.h"
#include "ntkernel.h"
#include "win32_message_text.h"

namespace {

inline constexpr ::std::io_scatter_t
com_name_message_range(::std::error_reporter_encoding encoding,
                       ::std::size_t startpos, ::std::size_t n) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    return {&startpos["\xC3\x96\x94\x93\xBD"], n}; // 5 bytes
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    return {&startpos[u"[com]"], n * sizeof(char16_t)};
  }
  case ::std::error_reporter_encoding::utf32: {
    return {&startpos[U"[com]"], n * sizeof(char32_t)};
  }
  default: {
    return {&startpos[u8"[com]"], n};
  }
  }
}

/*
Writes "(0x<hex HRESULT>)" for the requested encoding into __numbuf and
returns it as a scatter. The buffer must be at least
__format_hex_value_max_size_with_brackets<::std::uint_least32_t> code units
wide, each of the largest supported character size.
*/
inline constexpr ::std::io_scatter_t
__com_code_scatter(::std::error_reporter_encoding encoding,
                   ::std::uint_least32_t hr, char unsigned *__numbuf) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    auto *__dest{::std::error_domains::__herbceptions_detail::
                     __format_hex_value_full_with_bracket<true, char unsigned>(
                         __numbuf, hr)};
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
                reinterpret_cast<__char16_may_alias_ptr>(__numbuf), hr)};
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
                reinterpret_cast<__char32_may_alias_ptr>(__numbuf), hr)};
    return {__numbuf,
            static_cast<::std::size_t>(
                reinterpret_cast<char unsigned *>(__dest) - __numbuf)};
  }
  default: {
    auto *__dest{::std::error_domains::__herbceptions_detail::
                     __format_hex_value_full_with_bracket<false, char unsigned>(
                         __numbuf, hr)};
    return {__numbuf, static_cast<::std::size_t>(__dest - __numbuf)};
  }
  }
}

// HRESULT -> std::errc, mirroring status-code's com domain:
//   S_OK                          -> success
//   FACILITY_NT_BIT set           -> the embedded NTSTATUS via nt_errc_map
//   facility FACILITY_WIN32       -> the embedded Win32 code via win32_errc_map
//   anything else (E_FAIL & co)   -> io_error
inline constexpr ::std::errc com_to_errc(::std::uint_least32_t hr) noexcept {
  if (hr == 0) {
    return ::std::errc{};
  }
  if (hr & ::std::error_domains::__herbceptions_detail::__com_facility_nt_bit) {
    return ::std::error_domains::__herbceptions_detail::__nt_to_errc(
        hr &
        ~::std::error_domains::__herbceptions_detail::__com_facility_nt_bit);
  }
  if (::std::error_domains::__herbceptions_detail::__com_hresult_facility(hr) ==
      ::std::error_domains::__herbceptions_detail::__com_facility_win32) {
    return ::std::error_domains::__herbceptions_detail::__win32_to_errc(
        hr &
        ::std::error_domains::__herbceptions_detail::__com_hresult_code_mask);
  }
  return ::std::errc::io_error;
}

constinit ::std::error_domain_singleton com_error_domain{
    .do_cleanup = nullptr,
    .do_equivalent =
        [](::std::size_t cd, ::std::error_domain_singleton const *otherdomain,
           ::std::size_t othercd) noexcept {
          // com <-> com: identity.
          if (otherdomain == __builtin_addressof(com_error_domain))
            return cd == othercd;
          auto const hr{static_cast<::std::uint_least32_t>(cd)};
          // com (FACILITY_NT_BIT) <-> nt: exact match on the stripped
          // value. com (FACILITY_WIN32) <-> win32: exact match on
          // HRESULT_CODE. Other combinations compare through std::errc.
          ::std::int_least8_t rule{-1};
          if ((hr & ::std::error_domains::__herbceptions_detail::
                        __com_facility_nt_bit) != 0) {
            if (otherdomain == ::std::error_domains::__cxa_error_domain_nt()) {
              rule = ::std::error_domains::__herbceptions_detail::
                  __nt_com_equivalent(
                      static_cast<::std::uint_least32_t>(othercd), hr);
            }
          } else if (otherdomain ==
                     ::std::error_domains::__cxa_error_domain_win32()) {
            rule = ::std::error_domains::__herbceptions_detail::
                __com_win32_equivalent(
                    hr, static_cast<::std::uint_least32_t>(othercd));
          }
          if (rule >= 0)
            return rule == 1;
          return com_error_domain.do_to_errc(cd) ==
                 otherdomain->do_to_errc(othercd);
        },
    .do_query_information =
        [](::std::size_t cd, ::std::error_query_information query,
           ::std::error_reporter_encoding encoding, void *cookie,
           ::std::error_reporter_io_cookie_function cookfun) noexcept {
          // Windows DLLs provide nothing useful for plain HRESULTs
          // (E_FAIL & co), so no FormatMessage attempt is made here:
          //   FACILITY_NT_BIT codes render the ntkernel table's message for
          //   the embedded NTSTATUS; FACILITY_WIN32 codes delegate to the
          //   win32 domain, which renders the embedded Win32 code. Other
          //   facilities have no message at all.
          if (static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::name_message) <
              static_cast<::std::uint_least32_t>(query)) {
            return;
          }
          auto const hr{static_cast<::std::uint_least32_t>(cd)};
          if constexpr (::std::error_domains::__herbceptions_detail::
                            __is_freestanding_kernel_mode) {
            /*
            Freestanding kernel mode: the ntkernel table is a waste of rom
            size and delegation would pull in win32's machinery, so only
            [com](0x<hex code>) is ever reported. Only a small fixed
            scratch buffer is used.
            */
            ::std::io_scatter_t scatters[2];
            ::std::size_t scatterlen{};
            switch (query) {
            case ::std::error_query_information::name: {
              *scatters = com_name_message_range(encoding, 1u, 3u);
              scatterlen = 1u;
              break;
            }
            case ::std::error_query_information::message:
              [[fallthrough]];
            case ::std::error_query_information::name_message: {
              constexpr ::std::size_t bufbytes{
                  ::std::error_domains::__herbceptions_detail::
                      __format_hex_value_max_size_with_brackets<
                          ::std::uint_least32_t> *
                  sizeof(char32_t)};
              alignas(char32_t) char unsigned __numbuf[bufbytes];
              scatterlen = 0u;
              if (::std::error_query_information::name_message == query) {
                *scatters = com_name_message_range(encoding, 0u, 5u);
                ++scatterlen;
              }
              scatters[scatterlen] = __com_code_scatter(encoding, hr, __numbuf);
              ++scatterlen;
              break;
            }
            }
            cookfun(cookie, scatters, scatterlen);
          } else {
            ::std::io_scatter_t scatters[2];
            ::std::size_t scatterlen{};
            switch (query) {
            case ::std::error_query_information::name:
              *scatters = com_name_message_range(encoding, 1u, 3u);
              ++scatterlen;
              break;
            case ::std::error_query_information::message:
              [[fallthrough]];
            case ::std::error_query_information::name_message: {
              alignas(char32_t) char unsigned
                  __numbuf[::std::error_domains::__herbceptions_detail::
                               __format_hex_value_max_size_with_brackets<
                                   ::std::uint_least32_t> *
                           sizeof(char32_t)];
              if (::std::error_query_information::name_message == query) {
                *scatters = com_name_message_range(encoding, 0u, 5u);
                ++scatterlen;
              }
              scatters[scatterlen] = __com_code_scatter(encoding, hr, __numbuf);
              ++scatterlen;
              if (hr & ::std::error_domains::__herbceptions_detail::
                           __com_facility_nt_bit) {
                auto const msg{
                    ::std::error_domains::__herbceptions_detail::
                        __nt_u8_message(
                            hr & ~::std::error_domains::__herbceptions_detail::
                                     __com_facility_nt_bit)};
                if (msg.len != 0) {
                  char unsigned const *from_first{
                      reinterpret_cast<char unsigned const *>(msg.base)};
                  char unsigned const *from_last{from_first + msg.len};
                  alignas(char32_t) char unsigned
                      buffer[::std::error_domains::__herbceptions_detail::
                                 __nt_max_message_size *
                             sizeof(char32_t)];
                  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
                  case ::std::error_reporter_encoding::utfebcdic: {
                    auto dest{::std::error_domains::__herbceptions_detail::
                                  __write_ebcdic_with_ascii_only_range(
                                      from_first, from_last, buffer)};
                    scatters[scatterlen] = {
                        buffer, static_cast<::std::size_t>(dest - buffer)};
                    break;
                  }
#endif
                  case ::std::error_reporter_encoding::utf16: {
                    using __char16_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
                        [[__gnu__::__may_alias__]]
#endif
                        = char16_t *;
                    auto dest{::std::error_domains::__herbceptions_detail::
                                  __write_with_ascii_only_range(
                                      from_first, from_last,
                                      reinterpret_cast<__char16_may_alias_ptr>(
                                          buffer))};
                    scatters[scatterlen] = {
                        buffer,
                        static_cast<::std::size_t>(
                            reinterpret_cast<char unsigned *>(dest) - buffer)};
                    break;
                  }
                  case ::std::error_reporter_encoding::utf32: {
                    using __char32_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
                        [[__gnu__::__may_alias__]]
#endif
                        = char32_t *;
                    auto dest{::std::error_domains::__herbceptions_detail::
                                  __write_with_ascii_only_range(
                                      from_first, from_last,
                                      reinterpret_cast<__char32_may_alias_ptr>(
                                          buffer))};
                    scatters[scatterlen] = {
                        buffer,
                        static_cast<::std::size_t>(
                            reinterpret_cast<char unsigned *>(dest) - buffer)};
                    break;
                  }
                  default: {
                    scatters[scatterlen] = {
                        from_first,
                        static_cast<::std::size_t>(from_last - from_first)};
                    break;
                  }
                  }
                  ++scatterlen;
                }
                break;
              }
              if (::std::error_domains::__herbceptions_detail::
                      __com_hresult_facility(hr) ==
                  ::std::error_domains::__herbceptions_detail::
                      __com_facility_win32) {
                // Flush "[com](0x<hr>)" and then report only the
                // FormatMessage text for the embedded Win32 code; the win32
                // domain's own (0x<code>) block is deliberately not
                // repeated. The collector may be invoked several times;
                // each call appends.
                cookfun(cookie, scatters, scatterlen);
                ::std::error_domains::__herbceptions_detail::
                    __report_win32_message_text(
                        hr & ::std::error_domains::__herbceptions_detail::
                                 __com_hresult_code_mask,
                        encoding, cookie, cookfun);
                return;
              }
              break;
            }
            }
            cookfun(cookie, scatters, scatterlen);
          }
        },
    .do_to_errc = [](::std::size_t cd) noexcept -> ::std::errc {
      return com_to_errc(static_cast<::std::uint_least32_t>(cd));
    }};
} // namespace

extern "C" __HERBCEPTIONS_API ::std::error_domain_singleton const *
__cxa_error_domain_com() noexcept {
  return __builtin_addressof(com_error_domain);
}

#endif // _WIN32 || __CYGWIN__
