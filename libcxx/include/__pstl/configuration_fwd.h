//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CONFIGURATION_FWD_H
#define _LIBCPP___PSTL_CONFIGURATION_FWD_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <class... _Backends>
struct __backend_configuration;

struct __default_backend_tag;
struct __libdispatch_backend_tag;
struct __serial_backend_tag;
struct __std_thread_backend_tag;

#if defined(_LIBCPP_PSTL_BACKEND_SERIAL)
using __current_configuration = __backend_configuration<__serial_backend_tag, __default_backend_tag>;
#elif defined(_LIBCPP_PSTL_BACKEND_STD_THREAD)
using __current_configuration = __backend_configuration<__std_thread_backend_tag, __default_backend_tag>;
#elif defined(_LIBCPP_PSTL_BACKEND_LIBDISPATCH)
using __current_configuration = __backend_configuration<__libdispatch_backend_tag, __default_backend_tag>;
#else

// ...New vendors can add parallel backends here...

#  error "Invalid PSTL backend configuration"
#endif

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CONFIGURATION_FWD_H
