// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <__config>
#include <debugging>
#include <windows.h>

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

[[gnu::weak]] bool is_debugger_present() noexcept { return ::IsDebuggerPresent(); }

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS

_LIBCPP_END_NAMESPACE_STD
