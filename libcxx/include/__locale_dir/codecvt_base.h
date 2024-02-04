//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_CODECVT_BASE_H
#define _LIBCPP___LOCALE_CODECVT_BASE_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

class _LIBCPP_EXPORTED_FROM_ABI codecvt_base {
public:
  _LIBCPP_HIDE_FROM_ABI codecvt_base() {}
  enum result { ok, partial, error, noconv };
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_CODECVT_BASE_H
