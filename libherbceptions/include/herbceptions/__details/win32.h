//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
/*
win32 (win32_errc) error domain header.

Declares win32_errc (Win32 GetLastError codes, ERROR_*) and its error_domain
specialization. The singleton vtable is implemented in src/win32.cpp. Only
available on _WIN32/__CYGWIN__ targets.
*/

namespace std {

enum class win32_errc : ::std::uint_least32_t {};

inline constexpr bool operator==(::std::win32_errc __a,
                                 ::std::win32_errc __b) noexcept {
  return static_cast<::std::uint_least32_t>(__a) ==
         static_cast<::std::uint_least32_t>(__b);
}

template <> class error_domain<::std::win32_errc> {
public:
  using errc_type = ::std::win32_errc;
  static inline constexpr ::std::error_domain_singleton const *
  domain() noexcept {
    return ::std::error_domains::__cxa_error_domain_win32();
  }
  static inline constexpr ::std::size_t code(errc_type __e) noexcept {
    return static_cast<::std::size_t>(static_cast<::std::uint_least32_t>(__e));
  }
};

} // namespace std
