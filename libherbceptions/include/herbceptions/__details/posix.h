//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
/*
posix (std::errc) error domain header.

Declares the std::errc -> posix error_domain specialization. The singleton
vtable is implemented in src/posix.cpp.
*/

namespace std {
template <> class error_domain<::std::errc> {
public:
  using errc_type = ::std::errc;
  static inline constexpr ::std::error_domain_singleton const *
  domain() noexcept {
    return ::std::error_domains::__cxa_error_domain_posix();
  }
  static inline constexpr ::std::size_t code(errc_type __e) noexcept {
    return static_cast<::std::size_t>(
        static_cast<::std::underlying_type_t<errc_type>>(__e));
  }
};
} // namespace std
