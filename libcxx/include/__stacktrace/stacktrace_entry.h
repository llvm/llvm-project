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

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

#  include <__assert>
#  include <__functional/function.h>
#  include <__fwd/format.h>
#  include <__fwd/ostream.h>
#  include <__string/constexpr_c_functions.h>
#  include <cstddef>
#  include <cstdint>
#  include <optional>
#  include <string>

#  include <__stacktrace/string_manager.h>

_LIBCPP_BEGIN_NAMESPACE_STD

class stacktrace_entry;

namespace __stacktrace {

struct image;

struct entry_base {
  constexpr static size_t __max_sym_len = 512;
#  if defined(PATH_MAX)
  constexpr static size_t __max_file_len = PATH_MAX;
#  elif defined(MAX_PATH)
  constexpr static size_t __max_file_len = MAX_PATH;
#  else
  constexpr static size_t __max_file_len = (1 << 10);
#  endif

  uintptr_t __addr_{};
  optional<str> __desc_{};
  optional<str> __file_{};
  uint_least32_t __line_{};
  image* __image_{};

  void assign_desc(str __s) { __desc_ = std::move(__s); }
  void assign_file(str __s) { __file_ = std::move(__s); }

  _LIBCPP_EXPORTED_FROM_ABI std::ostream& write_to(std::ostream& __os) const;
  _LIBCPP_EXPORTED_FROM_ABI string to_string() const;
  _LIBCPP_HIDE_FROM_ABI uintptr_t adjusted_addr() const;
};

} // namespace __stacktrace

class stacktrace_entry : private __stacktrace::entry_base {
  __stacktrace::entry_base const& __base() const { return *(__stacktrace::entry_base const*)this; }
  friend _LIBCPP_EXPORTED_FROM_ABI inline ostream& operator<<(ostream& __os, std::stacktrace_entry const& __entry);
  friend _LIBCPP_EXPORTED_FROM_ABI inline string to_string(std::stacktrace_entry const& __entry);

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
    return __addr_;
  }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI constexpr explicit operator bool() const noexcept {
    return native_handle() != 0;
  }

  // (19.6.3.4) [stacktrace.entry.query], query
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI string description() const { return __desc_ ? string(*__desc_) : string{}; }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI string source_file() const { return __file_ ? string(*__file_) : string{}; }
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

_LIBCPP_EXPORTED_FROM_ABI inline ostream& operator<<(ostream& __os, std::stacktrace_entry const& __entry) {
  return __entry.__base().write_to(__os);
}

_LIBCPP_EXPORTED_FROM_ABI inline string to_string(std::stacktrace_entry const& __entry) {
  return __entry.__base().to_string();
}

// (19.6.5)
// Formatting support [stacktrace.format]:
// https://github.com/llvm/llvm-project/issues/105257

// (19.6.6)
// Hash support [stacktrace.basic.hash]

template <>
struct _LIBCPP_EXPORTED_FROM_ABI hash<stacktrace_entry> {
  [[nodiscard]] size_t operator()(stacktrace_entry const& __entry) const noexcept {
    auto __addr = __entry.native_handle();
    return hash<uintptr_t>()(__addr);
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_ENTRY
