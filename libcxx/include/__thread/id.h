// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___THREAD_ID_H
#define _LIBCPP___THREAD_ID_H

#include <__compare/ordering.h>
#include <__config>
#include <__fwd/functional.h>
#include <__fwd/ostream.h>
#include <__thread/support.h>
#include <__type_traits/conditional.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_pointer.h>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_HAS_THREADS
class __thread_id;

namespace this_thread {

_LIBCPP_HIDE_FROM_ABI __thread_id get_id() _NOEXCEPT;

} // namespace this_thread

template <>
struct hash<__thread_id>;

class __thread_id {
  __libcpp_thread_id __id_;

  // Even though __libcpp_thread_id is provided by underlying threading implementation
  // (e.g. C11, pthreads, or Windows) its type may still be unspecified. E.g. for pthreads
  // implementation __libcpp_thread_id is an alias for pthread_t, which is left unspecified
  // in POSIX. Typically it's either an integral type (glibc) or a pointer (Apple systems),
  // but it can also be an opaque type on some systems / libc implementations.
  //
  // Note that in order to satisfy standard requirements on std::thread::id, we need:
  // * strong total order
  // * formatter support
  // * std::hash implementation
  // Currently, we can implement all of the above only on pointer and integral types.
  using _Tp = __libcpp_thread_id;
  static_assert(is_pointer_v<_Tp> || is_integral_v<_Tp>, "unsupported thread::id type, please file a bug report");

  // Strong total order implementation.
  // Here we provide a best-effort implementation of strong total order, comparing
  // integral types as-is and routing pointers through uintptr_t for a well-defined comparison.
  static _LIBCPP_HIDE_FROM_ABI bool __eq_impl(__thread_id __x, __thread_id __y) _NOEXCEPT {
    if constexpr (is_pointer_v<_Tp>) {
      return reinterpret_cast<uintptr_t>(__x.__id_) == reinterpret_cast<uintptr_t>(__y.__id_);
    } else {
      return __x.__id_ == __y.__id_;
    }
  }

  static _LIBCPP_HIDE_FROM_ABI bool __lt_impl(__thread_id __x, __thread_id __y) _NOEXCEPT {
    if constexpr (is_pointer_v<_Tp>) {
      return reinterpret_cast<uintptr_t>(__x.__id_) < reinterpret_cast<uintptr_t>(__y.__id_);
    } else {
      // For integral thread IDs, assume 0 is always less than any other thread_id.
      if (__x.__id_ == 0)
        return __y.__id_ != 0;
      if (__y.__id_ == 0)
        return false;
      return __x.__id_ < __y.__id_;
    }
  }

  // Hashing implementation.
  // Simply use the underlying pointer or integral types as-is.
  using _HashTp = _Tp;
  _LIBCPP_HIDE_FROM_ABI _HashTp __hash_value() const { return __id_; }

  // Formatter implementation.
  // Note the output should match what the stream operator does. Since
  // the ostream operator has been shipped years before the formatter
  // was added to the Standard, our logic mimics what the stream
  // operator does (i.e. prints thread-id as integer, but uses a hexadecimal
  // format if it's represented by a pointer)
  using _FormatterTp = conditional_t<is_integral_v<_Tp>, _Tp, uintptr_t>;

  static _LIBCPP_HIDE_FROM_ABI constexpr bool __PRINT_AS_HEX = is_pointer_v<_Tp>;

  _LIBCPP_HIDE_FROM_ABI _FormatterTp __get_formatter_value() const { return reinterpret_cast<_FormatterTp>(__id_); }

public:
  _LIBCPP_HIDE_FROM_ABI __thread_id() _NOEXCEPT : __id_{} {}

  _LIBCPP_HIDE_FROM_ABI void __reset() { __id_ = __libcpp_thread_id{}; }

  friend _LIBCPP_HIDE_FROM_ABI bool operator==(__thread_id __x, __thread_id __y) _NOEXCEPT;
#  if _LIBCPP_STD_VER <= 17
  friend _LIBCPP_HIDE_FROM_ABI bool operator<(__thread_id __x, __thread_id __y) _NOEXCEPT;
#  else  // _LIBCPP_STD_VER <= 17
  friend _LIBCPP_HIDE_FROM_ABI strong_ordering operator<=>(__thread_id __x, __thread_id __y) noexcept;
#  endif // _LIBCPP_STD_VER <= 17

  template <class _CharT, class _Traits>
  friend _LIBCPP_HIDE_FROM_ABI basic_ostream<_CharT, _Traits>&
  operator<<(basic_ostream<_CharT, _Traits>& __os, __thread_id __id);

private:
  _LIBCPP_HIDE_FROM_ABI __thread_id(__libcpp_thread_id __id) : __id_{__id} {}

  friend __thread_id this_thread::get_id() _NOEXCEPT;
  friend class _LIBCPP_EXPORTED_FROM_ABI thread;
  friend struct hash<__thread_id>;

  template <class _Typename, class _CharT>
  friend struct formatter;
};

inline _LIBCPP_HIDE_FROM_ABI bool operator==(__thread_id __x, __thread_id __y) _NOEXCEPT {
  return __thread_id::__eq_impl(__x, __y);
}

#  if _LIBCPP_STD_VER <= 17

inline _LIBCPP_HIDE_FROM_ABI bool operator!=(__thread_id __x, __thread_id __y) _NOEXCEPT { return !(__x == __y); }

inline _LIBCPP_HIDE_FROM_ABI bool operator<(__thread_id __x, __thread_id __y) _NOEXCEPT {
  return __thread_id::__lt_impl(__x, __y);
}

inline _LIBCPP_HIDE_FROM_ABI bool operator<=(__thread_id __x, __thread_id __y) _NOEXCEPT { return !(__y < __x); }
inline _LIBCPP_HIDE_FROM_ABI bool operator>(__thread_id __x, __thread_id __y) _NOEXCEPT { return __y < __x; }
inline _LIBCPP_HIDE_FROM_ABI bool operator>=(__thread_id __x, __thread_id __y) _NOEXCEPT { return !(__x < __y); }

#  else // _LIBCPP_STD_VER <= 17

inline _LIBCPP_HIDE_FROM_ABI strong_ordering operator<=>(__thread_id __x, __thread_id __y) noexcept {
  if (__x == __y)
    return strong_ordering::equal;
  if (__thread_id::__lt_impl(__x, __y))
    return strong_ordering::less;
  return strong_ordering::greater;
}

#  endif // _LIBCPP_STD_VER <= 17

namespace this_thread {

inline _LIBCPP_HIDE_FROM_ABI __thread_id get_id() _NOEXCEPT { return __libcpp_thread_get_current_id(); }

} // namespace this_thread

#endif // _LIBCPP_HAS_THREADS

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___THREAD_ID_H
