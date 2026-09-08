//===--- nt.cpp - nt (nt_errc) error domain -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the nt (nt_errc, NTSTATUS STATUS_* codes) error_domain singleton
// vtable and the weak __cxa_error_domain_nt ABI entry point. Only built on
// _WIN32/__CYGWIN__ targets.
//
// The NTSTATUS -> {win32, posix, message} mapping is the ntkernel error
// category table (Apache-2.0 / Boost-1.0, Niall Douglas), embedded verbatim
// from ntkernel-table.ipp. All string literals are u8"" so they are UTF-8
// regardless of the execution character set.
//
// do_equivalent handles nt<->win32 via the table's win32 column, and
// nt<->posix (or any other domain) via the posix column.
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32) || defined(__CYGWIN__)

#include "libherbceptions.h"
#include "ntkernel.h"

namespace {

inline constexpr ::std::io_scatter_t
__nt_name_message_range(::std::error_reporter_encoding encoding,
                        ::std::size_t startpos, ::std::size_t n) noexcept {
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    return {&startpos["\xAD\x95\xA3\xBD"], n};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    return {&startpos[u"[nt]"], n * sizeof(char16_t)};
  }
  case ::std::error_reporter_encoding::utf32: {
    return {&startpos[U"[nt]"], n * sizeof(char32_t)};
  }
  default: {
    return {&startpos[u8"[nt]"], n};
  }
  }
}

inline constexpr ::std::io_scatter_t
__nt_name(::std::error_reporter_encoding encoding) noexcept {
  return __nt_name_message_range(encoding, 1u, 2u);
}

inline constexpr ::std::io_scatter_t
__nt_name_message(::std::error_reporter_encoding encoding) noexcept {
  return __nt_name_message_range(encoding, 0u, 4u);
}

// nt <-> win32 equivalence lives in ntkernel.h (__nt_win32_equivalent) so the
// win32 domain applies the identical rule.

// Writes "(0x<hex NTSTATUS>)" for the requested encoding into __numbuf and
// returns it as a scatter. The buffer must be at least
// __format_hex_value_max_size_with_brackets<::std::uint_least32_t> code units
// wide, each of the largest supported character size.
inline constexpr ::std::io_scatter_t
__nt_code_scatter(::std::error_reporter_encoding encoding,
                  ::std::uint_least32_t ntstatus,
                  char unsigned *__numbuf) noexcept {
  using namespace ::std::error_domains::__herbceptions_detail;
  switch (encoding) {
#ifdef __LIBHERBCEPTIONS_ENABLE_EBCDIC
  case ::std::error_reporter_encoding::utfebcdic: {
    auto *__dest{__format_hex_value_full_with_bracket<true, char unsigned>(
        __numbuf, ntstatus)};
    return {__numbuf, static_cast<::std::size_t>(__dest - __numbuf)};
  }
#endif
  case ::std::error_reporter_encoding::utf16: {
    using __char16_may_alias_ptr
#if __has_cpp_attribute(__gnu__::__may_alias__)
        [[__gnu__::__may_alias__]]
#endif
        = char16_t *;
    auto *__dest{__format_hex_value_full_with_bracket<false, char16_t>(
        reinterpret_cast<__char16_may_alias_ptr>(__numbuf), ntstatus)};
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
    auto *__dest{__format_hex_value_full_with_bracket<false, char32_t>(
        reinterpret_cast<__char32_may_alias_ptr>(__numbuf), ntstatus)};
    return {__numbuf,
            static_cast<::std::size_t>(
                reinterpret_cast<char unsigned *>(__dest) - __numbuf)};
  }
  default: {
    auto *__dest{__format_hex_value_full_with_bracket<false, char unsigned>(
        __numbuf, ntstatus)};
    return {__numbuf, static_cast<::std::size_t>(__dest - __numbuf)};
  }
  }
}

constinit ::std::error_domain_singleton nt_error_domain{
    .do_cleanup = nullptr,
    .do_equivalent =
        [](::std::size_t cd, ::std::error_domain_singleton const *otherdomain,
           ::std::size_t othercd) noexcept {
          // nt <-> nt: identity.
          if (otherdomain == __builtin_addressof(nt_error_domain))
            return cd == othercd;
          // nt <-> win32: use the table's win32 column.
          if (otherdomain == ::std::error_domains::__cxa_error_domain_win32())
            return ::std::error_domains::__herbceptions_detail::
                       __nt_win32_equivalent(
                           static_cast<::std::uint_least32_t>(cd),
                           static_cast<::std::uint_least32_t>(othercd)) == 1;
          // nt <-> com: an HRESULT carrying FACILITY_NT_BIT equates to the
          // embedded NTSTATUS exactly; other HRESULTs have no direct rule
          // and compare through std::errc below.
          if (otherdomain == ::std::error_domains::__cxa_error_domain_com()) {
            auto const rule{
                ::std::error_domains::__herbceptions_detail::
                    __nt_com_equivalent(
                        static_cast<::std::uint_least32_t>(cd),
                        static_cast<::std::uint_least32_t>(othercd))};
            if (rule >= 0)
              return rule == 1;
          }
          // nt <-> any other domain: compare via the POSIX errno mapping.
          return nt_error_domain.do_to_errc(static_cast<::std::uint_least32_t>(
                     cd)) == otherdomain->do_to_errc(othercd);
        },
    .do_query_information =
        [](::std::size_t cd, ::std::error_query_information query,
           ::std::error_reporter_encoding encoding, void *cookie,
           ::std::error_reporter_io_cookie_function cookfun) noexcept {
          // The message is the ntkernel table's descriptive string (u8"",
          // UTF-8 regardless of the execution character set). Windows DLLs
          // (FormatMessageW on ntdll) provide nothing useful for NTSTATUS,
          // so no fallback is attempted; codes absent from the table have
          // no message at all.
          if (static_cast<::std::uint_least32_t>(
                  ::std::error_query_information::name_message) <
              static_cast<::std::uint_least32_t>(query)) {
            return;
          }
          auto const ntstatus{static_cast<::std::uint_least32_t>(cd)};
          if constexpr (::std::error_domains::__herbceptions_detail::
                            __is_freestanding_kernel_mode) {
            /*
            Freestanding kernel mode: the ntkernel table is a waste of rom
            size, so only [nt](0x<hex code>) is ever reported and the table
            is never touched. Only a small fixed scratch buffer is used.
            */
            ::std::io_scatter_t scatters[2];
            ::std::size_t scatterlen{};
            switch (query) {
            case ::std::error_query_information::name: {
              *scatters = __nt_name(encoding);
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
                *scatters = __nt_name_message(encoding);
                ++scatterlen;
              }
              scatters[scatterlen] =
                  __nt_code_scatter(encoding, ntstatus, __numbuf);
              ++scatterlen;
              break;
            }
            }
            cookfun(cookie, scatters, scatterlen);
          } else {
            ::std::io_scatter_t scatters[3];
            ::std::size_t scatterlen{};
            switch (query) {
            case ::std::error_query_information::name: {
              *scatters = __nt_name(encoding);
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
                *scatters = __nt_name_message(encoding);
                ++scatterlen;
              }
              scatters[scatterlen] =
                  __nt_code_scatter(encoding, ntstatus, __numbuf);
              ++scatterlen;
              auto const msg{
                  ::std::error_domains::__herbceptions_detail::__nt_u8_message(
                      ntstatus)};
              if (msg.len == 0) {
                break;
              }
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
                auto dest{
                    ::std::error_domains::__herbceptions_detail::
                        __write_with_ascii_only_range(
                            from_first, from_last,
                            reinterpret_cast<__char16_may_alias_ptr>(buffer))};
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
                auto dest{
                    ::std::error_domains::__herbceptions_detail::
                        __write_with_ascii_only_range(
                            from_first, from_last,
                            reinterpret_cast<__char32_may_alias_ptr>(buffer))};
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
              break;
            }
            }
            cookfun(cookie, scatters, scatterlen);
          }
        },
    .do_to_errc = [](::std::size_t cd) noexcept -> ::std::errc {
      return ::std::error_domains::__herbceptions_detail::__nt_to_errc(
          static_cast<::std::uint_least32_t>(cd));
    }};
} // namespace

extern "C" __HERBCEPTIONS_API ::std::error_domain_singleton const *
__cxa_error_domain_nt() noexcept {
  return __builtin_addressof(nt_error_domain);
}

#endif // _WIN32 || __CYGWIN__
