//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
/*
Cross-domain mapping tables for the nt, win32 and com error domains
(private).

All lookup data lives in generated switch fragments (utils/ntkernel-table.json
is the source of truth, consumed by the generator that emits the .hpp files):
any nonzero NTSTATUS is treated as an error regardless of its severity bits.

The message switch is only ever referenced from hosted builds - freestanding
kernel translation units never emit it.
*/
#include <cstddef>
#include <cstdint>
#include <herbceptions/error>

namespace std::error_domains {
namespace __herbceptions_detail {

/*
COM HRESULT classification shared by the com, nt and win32 domains'
cross-domain equivalence rules.
*/
inline constexpr ::std::uint_least32_t __com_facility_nt_bit{0x10000000u};
inline constexpr ::std::uint_least32_t __com_facility_win32{7u};
inline constexpr ::std::uint_least32_t __com_hresult_code_mask{0xFFFFu};
inline constexpr ::std::uint_least32_t
__com_hresult_facility(::std::uint_least32_t hr) noexcept {
  return (hr >> 16) & 0x1FFFu;
}

// NTSTATUS -> equivalent Win32 GetLastError() code; 0 means none. The rule
// always applies: a 0 result definitively means "no equivalent".
inline ::std::uint_least32_t
__nt_to_win32_code(::std::uint_least32_t ntstatus) noexcept {
  switch (ntstatus) {
#include "nt_win32_map.hpp"
  }
  return static_cast<::std::uint_least32_t>(0);
}

// nt -> std::errc via the generated switch table (nt_errc_map.hpp, built by
// utils/generate_win32_nt_tables.py from the ntkernel-error-category and
// status-code tables). Only zero is success; every other NTSTATUS - positive
// or negative, whatever its severity bits say - is an error and must be
// mapped by the table, falling back to io_error when absent.
// Single definition of the NTSTATUS message switch (declared in
// ntkernel.h): keeps one copy of every message string in the binary.
inline constexpr ::std::errc __nt_to_errc(::std::uint_least32_t cd) noexcept {
  if (cd == 0)
    return ::std::errc{};
  switch (cd) {
#include "nt_errc_map.hpp"
  }
  return ::std::errc::io_error;
}

inline constexpr ::std::errc
__win32_to_errc(::std::uint_least32_t cd) noexcept {
  switch (static_cast<::std::uint_least32_t>(cd)) {
#include "win32_errc_map.hpp"
  }
  return ::std::errc::io_error;
}
// nt <-> win32 equivalence via the code maps:
//   1 equivalent / 0 definitely not equivalent / -1 unreachable here
// (the rule always applies, so it never reports "no opinion").
inline constexpr ::std::int_least8_t
__nt_win32_equivalent(::std::uint_least32_t ntstatus,
                      ::std::uint_least32_t win32err) noexcept {
  auto const mapped{__nt_to_win32_code(ntstatus)};
  if (mapped == 0) {
    return 0;
  }
  return mapped == win32err ? 1 : 0;
}

// nt <-> com: only an HRESULT carrying FACILITY_NT_BIT equates to the
// embedded NTSTATUS exactly; otherwise there is no specialized rule (-1).
inline constexpr ::std::int_least8_t
__nt_com_equivalent(::std::uint_least32_t ntstatus,
                    ::std::uint_least32_t hr) noexcept {
  if ((hr & __com_facility_nt_bit) != 0) {
    return (hr & ~__com_facility_nt_bit) == ntstatus ? 1 : 0;
  }
  return -1;
}

// com <-> win32: only a FACILITY_WIN32 HRESULT without the NT bit equates
// to its embedded Win32 code; otherwise no specialized rule (-1).
inline constexpr ::std::int_least8_t
__com_win32_equivalent(::std::uint_least32_t hr,
                       ::std::uint_least32_t win32err) noexcept {
  if ((hr & __com_facility_nt_bit) == 0 &&
      __com_hresult_facility(hr) == __com_facility_win32) {
    return (hr & __com_hresult_code_mask) == win32err ? 1 : 0;
  }
  return -1;
}

#include "nt_message_max.hpp"

// US-English UTF-8 message for an NTSTATUS; len == 0 when absent. Hosted
// builds only: never reference this from freestanding kernel translation
// units or the strings will be embedded into them. Defined in a single
// translation unit so the message table exists exactly once in the binary
// instead of once per including TU (GNU ld does not fold the copies).
inline constexpr ::std::io_scatter_t
__nt_u8_message(::std::uint_least32_t ntstatus) noexcept {
  switch (ntstatus) {
#include "nt_message_table.hpp"
  }
  return {nullptr, 0};
}

} // namespace __herbceptions_detail
} // namespace std::error_domains
