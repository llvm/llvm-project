//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBCPP_STACKTRACE_UNWIND_ADDRS_H
#define __LIBCPP_STACKTRACE_UNWIND_ADDRS_H

#include <__stacktrace/basic_stacktrace.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {
void unwind_addrs(std::__stacktrace::base& base, size_t skip, size_t depth);
} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#ifndef _WIN32

#  if __has_include(<libunwind.h>)
#    include <libunwind.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE inline void unwind_addrs(base& base, size_t skip, size_t depth) {
  if (!depth) {
    return;
  }
  unw_context_t cx;
  unw_getcontext(&cx);
  unw_cursor_t cur;
  unw_init_local(&cur, &cx);
  while (unw_step(&cur) > 0) {
    if (skip && skip--) {
      continue;
    }
    if (!depth--) {
      break;
    }
    auto& entry = base.__entry_append_();
    auto& eb    = (__stacktrace::entry_base&)entry;
    unw_get_reg(&cur, UNW_REG_IP, &eb.__addr_);
    if (!unw_is_signal_frame(&cur)) {
      --eb.__addr_;
    }
  }
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#  elif __has_include(<unwind.h>)
#    include <unwind.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct unwind_backtrace {
  base& base_;
  size_t skip_;
  size_t maxDepth_;

  _Unwind_Reason_Code callback(_Unwind_Context* ucx) {
    if (skip_) {
      --skip_;
      return _Unwind_Reason_Code::_URC_NO_REASON;
    }
    if (!maxDepth_) {
      return _Unwind_Reason_Code::_URC_NORMAL_STOP;
    }
    --maxDepth_;
    int ipBefore{0};
    auto ip = _Unwind_GetIPInfo(ucx, &ipBefore);
    if (!ip) {
      return _Unwind_Reason_Code::_URC_NORMAL_STOP;
    }
    auto& entry = base_.__entry_append_();
    auto& eb    = (entry_base&)entry;
    eb.__addr_  = (ipBefore ? ip : ip - 1);
    return _Unwind_Reason_Code::_URC_NO_REASON;
  }

  static _Unwind_Reason_Code callback(_Unwind_Context* cx, void* self) {
    return ((unwind_backtrace*)self)->callback(cx);
  }
};

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE inline void unwind_addrs(base& base, size_t skip, size_t depth) {
  if (!depth) {
    return;
  }
  unwind_backtrace bt{base, skip + 1, depth}; // skip this call as well
  _Unwind_Backtrace(unwind_backtrace::callback, &bt);
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#  else
#    error need <libunwind.h> or <unwind.h>
#  endif

#endif // _WIN32

#endif // __LIBCPP_STACKTRACE_UNWIND_ADDRS_H
