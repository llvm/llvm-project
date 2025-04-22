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

#include <__cstddef/byte.h>
#include <__cstddef/ptrdiff_t.h>
#include <__cstddef/size_t.h>
#include <__format/formatter.h>
#include <__functional/function.h>
#include <__functional/hash.h>
#include <__fwd/format.h>
#include <__fwd/ostream.h>
#include <__fwd/sstream.h>
#include <__fwd/vector.h>
#include <__iterator/iterator.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/reverse_access.h>
#include <__iterator/reverse_iterator.h>
#include <__memory/allocator.h>
#include <__memory/allocator_traits.h>
#include <__utility/move.h>
#include <__vector/pmr.h>
#include <__vector/swap.h>
#include <__vector/vector.h>
#include <cstdint>
#include <string>

#include <__stacktrace/entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD

class _LIBCPP_EXPORTED_FROM_ABI stacktrace_entry {
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
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI string description() const { return __desc_; }
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI string source_file() const { return __file_; }
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

  _LIBCPP_HIDE_FROM_ABI explicit stacktrace_entry(__stacktrace::entry&& __e)
      : __addr_(__e.__addr_), __desc_(std::move(__e.__desc_)), __file_(std::move(__e.__file_)), __line_(__e.__line_) {}

private:
  uintptr_t __addr_{};
  std::string __desc_{};
  std::string __file_{};
  uint_least32_t __line_{};
};

// (19.6.4.6)
// Non-member functions [stacktrace.basic.nonmem]

_LIBCPP_EXPORTED_FROM_ABI string to_string(const stacktrace_entry& __entry);
_LIBCPP_EXPORTED_FROM_ABI ostream& operator<<(ostream& __os, const stacktrace_entry& __entry);

// (19.6.5)
// Formatting support [stacktrace.format]

// TODO(stacktrace23): needs `formatter`
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

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_ENTRY
