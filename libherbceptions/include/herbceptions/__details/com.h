//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
/*
com (com_errc) error domain header.

Declares com_errc (HRESULT values) and its error_domain specialization. The
singleton vtable is implemented in src/com.cpp. Only available on
_WIN32/__CYGWIN__ targets.
*/

namespace std {

// The underlying type is uint32 so the standard 0x8000xxxx "failure" bit
// pattern fits without narrowing.
enum class com_errc : ::std::uint_least32_t {};

inline constexpr bool operator==(::std::com_errc __a,
                                 ::std::com_errc __b) noexcept {
  return static_cast<::std::uint_least32_t>(__a) ==
         static_cast<::std::uint_least32_t>(__b);
}

template <> class error_domain<::std::com_errc> {
public:
  using errc_type = ::std::com_errc;
  static inline constexpr ::std::error_domain_singleton const *
  domain() noexcept {
    return ::std::error_domains::__cxa_error_domain_com();
  }
  static inline constexpr ::std::size_t code(errc_type __e) noexcept {
    return static_cast<::std::size_t>(static_cast<::std::uint_least32_t>(__e));
  }
};

} // namespace std
