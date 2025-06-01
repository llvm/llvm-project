// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_ENTRY
#define _LIBCPP_STACKTRACE_ENTRY

#include <__config>
#include <__fwd/format.h>
#include <__fwd/ostream.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include <__stacktrace/base.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

class _LIBCPP_EXPORTED_FROM_ABI stacktrace_entry : private __stacktrace::entry_base {
  friend struct __stacktrace::entry_base;
  stacktrace_entry(entry_base const& __base) : entry_base(__base) {}

public:
  // (19.6.3.1) Overview [stacktrace.entry.overview]
  using native_handle_type = uintptr_t;

  // (19.6.3.2) [stacktrace.entry.cons], constructors
  _LIBCPP_EXPORTED_FROM_ABI ~stacktrace_entry() noexcept                                            = default;
  _LIBCPP_EXPORTED_FROM_ABI constexpr stacktrace_entry() noexcept                                   = default;
  _LIBCPP_EXPORTED_FROM_ABI constexpr stacktrace_entry(const stacktrace_entry&) noexcept            = default;
  _LIBCPP_EXPORTED_FROM_ABI constexpr stacktrace_entry& operator=(const stacktrace_entry&) noexcept = default;

  // (19.6.3.3) [stacktrace.entry.obs], observers
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI constexpr native_handle_type native_handle() const noexcept {
    return __addr_actual_;
  }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI constexpr explicit operator bool() const noexcept {
    return native_handle() != 0;
  }

  // (19.6.3.4) [stacktrace.entry.query], query
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI string description() const {
    if (__desc_->empty()) {
      return "";
    }
    return {__desc_->data(), __desc_->size()};
  }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI string source_file() const {
    if (__desc_->empty()) {
      return "";
    }
    return {__file_->data(), __file_->size()};
  }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI uint_least32_t source_line() const { return __line_; }

  // (19.6.3.5) [stacktrace.entry.cmp], comparison
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI friend constexpr bool
  operator==(const stacktrace_entry& __x, const stacktrace_entry& __y) noexcept {
    return __x.native_handle() == __y.native_handle();
  }

  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI friend constexpr strong_ordering
  operator<=>(const stacktrace_entry& __x, const stacktrace_entry& __y) noexcept {
    return __x.native_handle() <=> __y.native_handle();
  }
};

// (19.6.4.6)
// Non-member functions [stacktrace.basic.nonmem]

[[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI string to_string(const stacktrace_entry& __entry);
_LIBCPP_EXPORTED_FROM_ABI ostream& operator<<(ostream& __os, const stacktrace_entry& __entry);

// (19.6.5)
// Formatting support [stacktrace.format]

// TODO: stacktrace formatter: https://github.com/llvm/llvm-project/issues/105257
template <>
struct _LIBCPP_EXPORTED_FROM_ABI formatter<stacktrace_entry>;

// (19.6.6)
// Hash support [stacktrace.basic.hash]

template <>
struct _LIBCPP_EXPORTED_FROM_ABI hash<stacktrace_entry> {
  [[nodiscard]] size_t operator()(stacktrace_entry const& __entry) const noexcept {
    auto __addr = __entry.native_handle();
    return hash<uintptr_t>()(__addr);
  }
};

namespace __stacktrace {
inline stacktrace_entry entry_base::to_stacktrace_entry() const { return {*this}; }
} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_ENTRY
