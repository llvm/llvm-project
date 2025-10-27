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

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#include <__assert>
#include <__functional/function.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#if _LIBCPP_HAS_LOCALIZATION
#  include <__fwd/format.h>
#  include <__fwd/ostream.h>
#endif // _LIBCPP_HAS_LOCALIZATION

#if _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

_LIBCPP_BEGIN_NAMESPACE_STD

class stacktrace_entry;

namespace __stacktrace {

struct _Str;

template <class _Tp>
struct _Str_Alloc {
  using result_t _LIBCPP_NODEBUG = allocation_result<_Tp*, size_t>;

  // Lambdas wrap the caller's allocator, re-bound so we can deal with `chars`.
  function<char*(size_t)> __alloc_;
  function<void(char*, size_t)> __dealloc_;

  // This only works with chars or other 1-byte things.
  static_assert(sizeof(_Tp) == 1);

  using value_type = _Tp;
  using pointer    = _Tp*;

  template <class _Tp2>
  struct rebind {
    using other = _Str_Alloc<_Tp2>;
  };

  _Str_Alloc(const _Str_Alloc&)            = default;
  _Str_Alloc(_Str_Alloc&&)                 = default;
  _Str_Alloc& operator=(const _Str_Alloc&) = default;
  _Str_Alloc& operator=(_Str_Alloc&&)      = default;

  _Str_Alloc(function<char*(size_t)> __alloc, function<void(char*, size_t)> __dealloc)
      : __alloc_(std::move(__alloc)), __dealloc_(std::move(__dealloc)) {}

  template <class _A0, // some allocator; can be of any type
            class _AT = allocator_traits<_A0>,
            class _CA = typename _AT::template rebind_alloc<char>>
    requires __is_allocator_v<_A0>
  static _Str_Alloc make(_A0 __a) {
    auto __ca = _CA(__a);
    return {[__ca](size_t __n) mutable -> char* { return __ca.allocate(__n); },
            [__ca](char* __p, size_t __n) mutable { __ca.deallocate(__p, __n); }};
  }

  _Tp* allocate(size_t __n) { return __alloc_(__n); }
  void deallocate(_Tp* __p, size_t __n) { __dealloc_(__p, __n); }
  bool operator==(_Str_Alloc<_Tp> const& __rhs) const { return std::addressof(__rhs) == this; }
};

struct _Str : basic_string<char, char_traits<char>, _Str_Alloc<char>> {
  using base = basic_string<char, char_traits<char>, _Str_Alloc<char>>;
  _LIBCPP_HIDE_FROM_ABI _Str(_Str_Alloc<char> const& __alloc) : base(__alloc) {}
  _LIBCPP_HIDE_FROM_ABI string_view view() const { return {this->data(), this->size()}; }
};

struct _Image;

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
  optional<_Str> __desc_{};
  optional<_Str> __file_{};
  uint_least32_t __line_{};
  _Image const* __image_{};

  _LIBCPP_HIDE_FROM_ABI _Str& assign_desc(_Str&& __s) { return *(__desc_ = std::move(__s)); }
  _LIBCPP_HIDE_FROM_ABI _Str& assign_file(_Str&& __s) { return *(__file_ = std::move(__s)); }

#  if _LIBCPP_HAS_LOCALIZATION
  _LIBCPP_EXPORTED_FROM_ABI std::ostream& write_to(std::ostream& __os) const;
  _LIBCPP_EXPORTED_FROM_ABI string to_string() const;
#  endif // _LIBCPP_HAS_LOCALIZATION

  _LIBCPP_HIDE_FROM_ABI uintptr_t adjusted_addr() const;

  _LIBCPP_HIDE_FROM_ABI constexpr static _Entry* of(auto& __s) { return static_cast<_Entry*>(__s); }

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
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI string description() const {
    return __base_.__desc_ ? string(*__base_.__desc_) : string();
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI string source_file() const {
    return __base_.__file_ ? string(*__base_.__file_) : string();
  }
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
