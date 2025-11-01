// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_ENTRY_H
#define _LIBCPP_STACKTRACE_ENTRY_H

#include <__config>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#include <__assert>
#include <__functional/function.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#if _LIBCPP_HAS_LOCALIZATION
#  include <__fwd/format.h>
#  include <__fwd/ostream.h>
#endif // _LIBCPP_HAS_LOCALIZATION

#if _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

_LIBCPP_BEGIN_NAMESPACE_STD

class stacktrace_entry;

namespace __stacktrace {

struct _Image;

struct _StringWrapper {
  // XXX FIXME TODO:
  // Figure out a solution for creating strings while respecting
  // the caller's allocator they provided:
  //   1. properly type-erase basic_strings' allocator types
  //   2. move all code into headers (seems like a bad idea)
  //   3. leave these as oversized char arrays, seems suboptimal
  //   4. just use std::string, which is just plain wrong
  //   5. ...?

  char __chars_[1024];

  _LIBCPP_HIDE_FROM_ABI std::string_view view() const { return __chars_; }

  _LIBCPP_HIDE_FROM_ABI _StringWrapper& assign(std::string_view __view) {
    size_t __size = std::min(__view.size(), sizeof(__chars_) - 1);
    memcpy(__chars_, __view.data(), __size);
    __chars_[__size] = 0;
    return *this;
  }
};

struct _Entry {
  constexpr static size_t __max_sym_len = 512;
#  if defined(PATH_MAX)
  constexpr static size_t __max_file_len = PATH_MAX;
#  elif defined(MAX_PATH)
  constexpr static size_t __max_file_len = MAX_PATH;
#  else
  constexpr static size_t __max_file_len = (1 << 10);
#  endif

  uintptr_t __addr_{};
  _StringWrapper __desc_{};
  _StringWrapper __file_{};
  uint_least32_t __line_{};
  _Image const* __image_{};

#  if _LIBCPP_HAS_LOCALIZATION
  _LIBCPP_EXPORTED_FROM_ABI std::ostream& write_to(std::ostream& __os) const;
  _LIBCPP_EXPORTED_FROM_ABI string to_string() const;
#  endif // _LIBCPP_HAS_LOCALIZATION

  _LIBCPP_HIDE_FROM_ABI uintptr_t adjusted_addr() const;

  _LIBCPP_HIDE_FROM_ABI ~_Entry()                                  = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry()                         = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry(const _Entry&)            = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry& operator=(const _Entry&) = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry(_Entry&&)                 = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry& operator=(_Entry&&)      = default;
};

} // namespace __stacktrace

class stacktrace_entry {
  friend _LIBCPP_HIDE_FROM_ABI inline ostream& operator<<(ostream& __os, std::stacktrace_entry const& __entry);
  friend _LIBCPP_HIDE_FROM_ABI inline string to_string(std::stacktrace_entry const& __entry);

  __stacktrace::_Entry __base_{};

public:
  // (19.6.3.1) Overview [stacktrace.entry.overview]
  using native_handle_type = uintptr_t;

  // (19.6.3.2) [stacktrace.entry.cons], constructors
  _LIBCPP_HIDE_FROM_ABI ~stacktrace_entry() noexcept                                            = default;
  _LIBCPP_HIDE_FROM_ABI constexpr stacktrace_entry() noexcept                                   = default;
  _LIBCPP_HIDE_FROM_ABI constexpr stacktrace_entry(const stacktrace_entry&) noexcept            = default;
  _LIBCPP_HIDE_FROM_ABI constexpr stacktrace_entry& operator=(const stacktrace_entry&) noexcept = default;

  // (19.6.3.3) [stacktrace.entry.obs], observers
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr native_handle_type native_handle() const noexcept {
    return __base_.__addr_;
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr explicit operator bool() const noexcept { return native_handle() != 0; }

  // (19.6.3.4) [stacktrace.entry.query], query
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI string description() const { return string(__base_.__desc_.view()); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI string source_file() const { return string(__base_.__file_.view()); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI uint_least32_t source_line() const { return __base_.__line_; }

  // (19.6.3.5) [stacktrace.entry.cmp], comparison
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const stacktrace_entry& __x, const stacktrace_entry& __y) noexcept {
    return __x.native_handle() == __y.native_handle();
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI friend constexpr strong_ordering
  operator<=>(const stacktrace_entry& __x, const stacktrace_entry& __y) noexcept {
    return __x.native_handle() <=> __y.native_handle();
  }
};

// (19.6.4.6)
// Non-member functions [stacktrace.basic.nonmem]

[[nodiscard]] _LIBCPP_HIDE_FROM_ABI string to_string(const stacktrace_entry& __entry);

#  if _LIBCPP_HAS_LOCALIZATION
_LIBCPP_HIDE_FROM_ABI inline ostream& operator<<(ostream& __os, std::stacktrace_entry const& __entry) {
  return __entry.__base_.write_to(__os);
}
_LIBCPP_HIDE_FROM_ABI inline string to_string(std::stacktrace_entry const& __entry) {
  return __entry.__base_.to_string();
}
#  endif // _LIBCPP_HAS_LOCALIZATION

// (19.6.5)
// Formatting support [stacktrace.format]:
// https://github.com/llvm/llvm-project/issues/105257

// (19.6.6)
// Hash support [stacktrace.basic.hash]

template <>
struct _LIBCPP_HIDE_FROM_ABI hash<stacktrace_entry> {
  [[nodiscard]] size_t operator()(stacktrace_entry const& __entry) const noexcept {
    auto __addr = __entry.native_handle();
    return hash<uintptr_t>()(__addr);
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_ENTRY_H
