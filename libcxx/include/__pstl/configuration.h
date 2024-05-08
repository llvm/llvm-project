//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CONFIGURATION_H
#define _LIBCPP___PSTL_CONFIGURATION_H

#include <__config>
#include <__pstl/configuration_fwd.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if defined(_LIBCPP_PSTL_BACKEND_SERIAL)
#  include <__pstl/backends/serial.h>
#elif defined(_LIBCPP_PSTL_BACKEND_STD_THREAD)
#  include <__pstl/backends/std_thread.h>
#elif defined(_LIBCPP_PSTL_BACKEND_LIBDISPATCH)
#  include <__pstl/backends/libdispatch.h>
#endif

#endif // _LIBCPP___PSTL_CONFIGURATION_H
