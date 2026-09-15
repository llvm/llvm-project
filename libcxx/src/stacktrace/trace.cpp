//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__stacktrace/basic_stacktrace.h>
#include <__stacktrace/stacktrace_entry.h>
#include <stacktrace>
#include <string>

#if _LIBCPP_HAS_LOCALIZATION
#  include <iostream>
#  include <sstream>
#endif //_LIBCPP_HAS_LOCALIZATION

#if !defined(_WIN32)
#  include <unwind.h>
#endif

// Purposely avoids optimizations to make call-chain predictable
#if __has_cpp_attribute(_Clang::__disable_tail_calls__)
#  define _LIBCPP_STACKTRACE_NO_TAIL_CALLS_OUT [[_Clang::__disable_tail_calls__]]
#elif __has_cpp_attribute(__gnu__::__optimize__)
#  define _LIBCPP_STACKTRACE_NO_TAIL_CALLS_OUT [[__gnu__::__optimize__("no-optimize-sibling-calls")]]
#else
#  define _LIBCPP_STACKTRACE_NO_TAIL_CALLS_OUT
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace __stacktrace {

#if !defined(_WIN32)

namespace {

struct _Unwind_Wrapper {
  _Trace& base_;
  size_t skip_;
  size_t maxDepth_;

  _Unwind_Reason_Code callback(_Unwind_Context* __ucx) {
    if (skip_) {
      --skip_;
      return _Unwind_Reason_Code::_URC_NO_REASON;
    }
    if (!maxDepth_) {
      return _Unwind_Reason_Code::_URC_NORMAL_STOP;
    }
    --maxDepth_;
    int __ip_before{0};
    auto __ip = _Unwind_GetIPInfo(__ucx, &__ip_before);
    if (!__ip) {
      return _Unwind_Reason_Code::_URC_NORMAL_STOP;
    }
    auto& __entry = base_.__entry_append_();
    auto& __eb    = (_Entry&)__entry;
    __eb.__addr_  = (__ip_before ? __ip : __ip - 1);
    return _Unwind_Reason_Code::_URC_NO_REASON;
  }

  static _Unwind_Reason_Code callback(_Unwind_Context* __cx, void* __self) {
    return ((_Unwind_Wrapper*)__self)->callback(__cx);
  }
};

} // namespace

// Kept out-of-line here rather than in the header: GCC has been observed to inline this
// despite `noinline` when reached through an always-inline call chain with a single call
// site, which silently shifts the captured frames by one (see the `+1` below).
_LIBCPP_STACKTRACE_NO_TAIL_CALLS_OUT void _Trace::__populate_addrs(size_t __skip, size_t __depth) {
  if (!__depth) {
    return;
  }
  _Unwind_Wrapper __bt{*this, __skip + 1, __depth}; /* +1 to skip our own frame */
  _Unwind_Backtrace(_Unwind_Wrapper::callback, &__bt);
}

#endif // !_WIN32

#if _LIBCPP_HAS_LOCALIZATION

ostream& _Trace::__write_to(std::ostream& __os) const {
  auto iters = __entry_iters_();
  auto count = iters.size();
  if (!count) {
    __os << "(empty stacktrace)";
  } else {
    for (size_t __i = 0; __i < count; __i++) {
      // Insert newlines between entries (but not before the first or after the last)
      if (__i) {
        __os << '\n';
      }

      stacktrace_entry& entry = *reinterpret_cast<stacktrace_entry*>(iters.data() + __i);

      // printf-style format to a small buffer, to avoid messing with stream (with `setw` etc.)
      char index_str[21];
      snprintf(index_str, sizeof(index_str), "%3zu", __i + 1);
      __os << "  frame " << index_str << ": " << entry;
    }
  }
  return __os;
}

string _Trace::__to_string() const {
  stringstream __ss;
  __write_to(__ss);
  return __ss.str();
}

#endif // _LIBCPP_HAS_LOCALIZATION

size_t _Trace::__hash_code() const {
  size_t __ret = size_t(0xc3a5c85c97cb3127ull); // taken from __functional/hash.h
  for (_Entry const& __e : __entry_iters_()) {
    __ret = (__ret << 1) ^ __e.__hash_code();
  }
  return __ret;
}

} // namespace __stacktrace

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
