//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "stacktrace/config.h"

#if defined(_LIBCPP_STACKTRACE_COLLECT_UNWIND)

#  include <unwind.h>

#  include "stacktrace/unwind.h"
#  include <__stacktrace/basic_stacktrace.h>
#  include <__stacktrace/stacktrace_entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct unwind_backtrace {
  context& cx_;
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
    int ipBefore;
    auto ip = _Unwind_GetIPInfo(ucx, &ipBefore);
    if (!ip) {
      return _Unwind_Reason_Code::_URC_NORMAL_STOP;
    }
    auto& entry          = cx_.__entries_.emplace_back();
    entry.__addr_        = (ipBefore ? ip : ip - 1);
    entry.__addr_unslid_ = entry.__addr_; // in case we can't un-slide
    return _Unwind_Reason_Code::_URC_NO_REASON;
  }

  static _Unwind_Reason_Code callback(_Unwind_Context* cx, void* self) {
    return ((unwind_backtrace*)self)->callback(cx);
  }
};

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void unwind::collect(size_t skip, size_t max_depth) {
  unwind_backtrace bt{cx_, skip + 1, max_depth}; // skip this call as well
  _Unwind_Backtrace(unwind_backtrace::callback, &bt);
}

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif
