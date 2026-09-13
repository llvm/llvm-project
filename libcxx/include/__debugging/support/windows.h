// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___DEBUGGING_SUPPORT_WINDOWS_H
#define _LIBCPP___DEBUGGING_SUPPORT_WINDOWS_H

#include <__config>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

_LIBCPP_HIDE_FROM_ABI inline bool __libcpp_is_debugger_present() noexcept { return ::IsDebuggerPresent(); }

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___DEBUGGING_SUPPORT_WINDOWS_H
