// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___STACKTRACE_ENTRY_H
#define _LIBCPP___STACKTRACE_ENTRY_H

#include <__assert>
#include <__config>
#include <__functional/function.h>
#include <__fwd/format.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#if _LIBCPP_HAS_LOCALIZATION
#  include <__fwd/ostream.h>
#endif // _LIBCPP_HAS_LOCALIZATION

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

_LIBCPP_BEGIN_NAMESPACE_STD

class stacktrace_entry;

namespace __stacktrace {

struct _Image;

struct _StringWrapper {
  // XXX FIXME TODO:
  // Figure out a solution for creating strings while respecting
  // the caller's allocator they provided:
  //   1. properly typeerase basic_strings' allocator types
  //   2. move all code into headers (seems like a bad idea)
  //   3. leave these as oversized char arrays, seems suboptimal
  //   4. just use std::string, which is just plain wrong
  //   5. ...?

  std::string __str_;

  _LIBCPP_HIDE_FROM_ABI std::string_view view() const { return __str_; }

  _LIBCPP_HIDE_FROM_ABI _StringWrapper& assign(std::string_view __view) {
    __str_ = __view;
    return *this;
  }
};

struct _Entry {
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

  _LIBCPP_EXPORTED_FROM_ABI size_t hash() const;
  _LIBCPP_HIDE_FROM_ABI static _Entry& base(stacktrace_entry& __entry);
  _LIBCPP_HIDE_FROM_ABI static _Entry const& base(stacktrace_entry const& __entry);

  _LIBCPP_HIDE_FROM_ABI uintptr_t adjusted_addr() const;

  _LIBCPP_HIDE_FROM_ABI ~_Entry()                                  = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry()                         = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry(const _Entry&)            = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry& operator=(const _Entry&) = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry(_Entry&&)                 = default;
  _LIBCPP_HIDE_FROM_ABI constexpr _Entry& operator=(_Entry&&)      = default;
};

struct _Trace;

} // namespace __stacktrace

class stacktrace_entry {
  _LIBCPP_HIDE_FROM_ABI friend struct __stacktrace::_Entry;
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
  _LIBCPP_HIDE_FROM_ABI constexpr native_handle_type native_handle() const noexcept { return __base_.__addr_; }
  _LIBCPP_HIDE_FROM_ABI constexpr explicit operator bool() const noexcept { return native_handle() != 0; }

  // (19.6.3.4) [stacktrace.entry.query], query
  _LIBCPP_HIDE_FROM_ABI string description() const { return string(__base_.__desc_.view()); }
  _LIBCPP_HIDE_FROM_ABI string source_file() const { return string(__base_.__file_.view()); }
  _LIBCPP_HIDE_FROM_ABI uint_least32_t source_line() const { return __base_.__line_; }

  // (19.6.3.5) [stacktrace.entry.cmp], comparison
  _LIBCPP_HIDE_FROM_ABI friend constexpr bool
  operator==(const stacktrace_entry& __x, const stacktrace_entry& __y) noexcept {
    return __x.native_handle() == __y.native_handle();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr strong_ordering
  operator<=>(const stacktrace_entry& __x, const stacktrace_entry& __y) noexcept {
    return __x.native_handle() <=> __y.native_handle();
  }
};

// (19.6.4.6)
// Non-member functions [stacktrace.basic.nonmem]

#  if _LIBCPP_HAS_LOCALIZATION

_LIBCPP_HIDE_FROM_ABI inline string to_string(const std::stacktrace_entry& __entry) {
  return __stacktrace::_Entry::base(__entry).to_string();
}

_LIBCPP_HIDE_FROM_ABI inline ostream& operator<<(ostream& __os, const stacktrace_entry& __entry) {
  return __stacktrace::_Entry::base(__entry).write_to(__os);
}

#  endif // _LIBCPP_HAS_LOCALIZATION

// (19.6.5)
// Formatting support [stacktrace.format]:
// https://github.com/llvm/llvm-project/issues/105257

// (19.6.6)
// Hash support [stacktrace.basic.hash]

template <>
struct hash<stacktrace_entry> {
  _LIBCPP_HIDE_FROM_ABI size_t operator()(const stacktrace_entry& __entry) const noexcept {
    return __stacktrace::_Entry::base(__entry).hash();
  }
};

namespace __stacktrace {

_LIBCPP_HIDE_FROM_ABI inline _Entry& _Entry::base(stacktrace_entry& __entry) { return __entry.__base_; }

_LIBCPP_HIDE_FROM_ABI inline _Entry const& _Entry::base(stacktrace_entry const& __entry) { return __entry.__base_; }

} // namespace __stacktrace

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

_LIBCPP_POP_MACROS

#endif // _LIBCPP___STACKTRACE_ENTRY_H
