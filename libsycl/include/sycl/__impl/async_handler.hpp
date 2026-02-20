//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL async_handler type, which
/// is a callable such as a function class or lambda, with an exception_list as
/// a parameter. Invocation of an async_handler may be triggered by the queue
/// member functions queue::wait_and_throw or queue::throw_asynchronous, by the
/// event member function event::wait_and_throw, or automatically on destruction
/// of a queue or context that contains unconsumed asynchronous errors.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_ASYNC_HANDLER_HPP
#define _LIBSYCL___IMPL_ASYNC_HANDLER_HPP

#include <functional>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class exception_list;

// SYCL 2020 4.13.2. Exception class interface.
using async_handler = std::function<void(sycl::exception_list)>;

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_ASYNC_HANDLER_HPP
