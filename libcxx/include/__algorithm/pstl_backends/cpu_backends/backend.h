//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_BACKEND_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_BACKEND_H

#include <__config>

#if defined(_LIBCPP_PSTL_CPU_BACKEND_SERIAL)
#  include <__algorithm/pstl_backends/cpu_backends/serial.h>
#elif defined(_LIBCPP_PSTL_CPU_BACKEND_THREAD)
#  include <__algorithm/pstl_backends/cpu_backends/thread.h>
#else
#  error "Invalid CPU backend choice"
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

struct __cpu_backend_tag {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_CPU_BACKEND_BACKEND_H
