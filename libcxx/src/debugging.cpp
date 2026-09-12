//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__assert>
#include <__config>
#include <__debugging/support.h>
#include <debugging>

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

[[gnu::weak]] bool is_debugger_present() noexcept { return __libcpp_is_debugger_present(); }

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS

_LIBCPP_END_NAMESPACE_STD
