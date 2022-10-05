// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHRONO_OSTREAM_H
#define _LIBCPP___CHRONO_OSTREAM_H

#include <__chrono/day.h>
#include <__chrono/statically_widen.h>
#include <__chrono/year.h>
#include <__config>
#include <__format/format_functions.h>
#include <ostream>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_FORMAT)

namespace chrono {

template <class _CharT, class _Traits>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_AVAILABILITY_FORMAT basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, const day& __d) {
  return __os
      << (__d.ok()
              ? std::format(_LIBCPP_STATICALLY_WIDEN(_CharT, "{:%d}"), __d)
              // Note this error differs from the wording of the Standard. The
              // Standard wording doesn't work well on AIX or Windows. There
              // the formatted day seems to be either modulo 100 or completely
              // omitted. Judging by the wording this is valid.
              // TODO FMT Write a paper of file an LWG issue.
              : std::format(_LIBCPP_STATICALLY_WIDEN(_CharT, "{:02} is not a valid day"), static_cast<unsigned>(__d)));
}

template <class _CharT, class _Traits>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_AVAILABILITY_FORMAT basic_ostream<_CharT, _Traits>&
operator<<(basic_ostream<_CharT, _Traits>& __os, const year& __y) {
  return __os << (__y.ok() ? std::format(_LIBCPP_STATICALLY_WIDEN(_CharT, "{:%Y}"), __y)
                           : std::format(_LIBCPP_STATICALLY_WIDEN(_CharT, "{:%Y} is not a valid year"), __y));
}

} // namespace chrono

#endif //if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_FORMAT)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CHRONO_OSTREAM_H
