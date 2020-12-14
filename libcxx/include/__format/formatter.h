// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_FORMATTER_H
#define _LIBCPP___FORMAT_FORMATTER_H

#include <__availability>
#include <__config>
#include <__format/format_error.h>
#include <__format/parser_std_format_spec.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17

// TODO FMT Remove this once we require compilers with proper C++20 support.
// If the compiler has no concepts support, the format header will be disabled.
// Without concepts support enable_if needs to be used and that too much effort
// to support compilers with partial C++20 support.
#if !defined(_LIBCPP_HAS_NO_CONCEPTS)

// Currently not implemented specializations throw an exception when used. This
// does not conform to the Standard. However not all Standard defined formatters
// have been implemented yet. Until that time the current behavior is intended.
// TODO FMT Disable the default template.
template <class _Tp, class _CharT>
struct _LIBCPP_TEMPLATE_VIS _LIBCPP_AVAILABILITY_FORMAT formatter {
  _LIBCPP_NORETURN _LIBCPP_HIDE_FROM_ABI auto parse(auto& __parse_ctx)
      -> decltype(__parse_ctx.begin()) {
    __throw();
  }

  _LIBCPP_NORETURN _LIBCPP_HIDE_FROM_ABI auto format(_Tp, auto& __ctx)
      -> decltype(__ctx.out()) {
    __throw();
  }

private:
  _LIBCPP_NORETURN _LIBCPP_HIDE_FROM_ABI void __throw() {
    __throw_format_error("Argument type not implemented yet");
  }
};

#endif // !defined(_LIBCPP_HAS_NO_CONCEPTS)

#endif //_LIBCPP_STD_VER > 17

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___FORMAT_FORMATTER_H
