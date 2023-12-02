//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_SHARED_RESOURCE_H
#define _LIBCPP___UTILITY_SHARED_RESOURCE_H

#include <__availability>
#include <__config>
#include <__mutex/lock_guard.h>
#include <__mutex/mutex.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#ifndef _LIBCPP_HAS_NO_THREADS

// Contains helper functions for a reference counted resource.
// The resources are identified by their address. The reference counting is
// done by calling increasing and decreasing the reference count at the
// appropriate time.
//
// There are two ways to guard the resource against concurrent access:
// - Call the lock function. The function locks two mutexes
//   - The internal map's mutex
//   - The mutex of the resource
//   This approach does not require to store a pointer/reference to the mutex,
//   reducing the size of the object.
// - Store the mutex of the inc function. Then the caller can lock the mutex at
//   the appropriate time.

_LIBCPP_BEGIN_NAMESPACE_STD

// std::mutex is available as an extension since C++03, but returning a
// lock_guard requires C++17. Since the current use-cases require newer C++
// versions the older versions are not supported.
#  if _LIBCPP_STD_VER >= 17

_LIBCPP_EXPORTED_FROM_ABI _LIBCPP_AVAILABILITY_SHARED_RESOURCE mutex&
__shared_resource_inc_reference(const void* __ptr);

// precondition: __ptr's reference count > 0
_LIBCPP_EXPORTED_FROM_ABI _LIBCPP_AVAILABILITY_SHARED_RESOURCE void __shared_resource_dec_reference(const void* __ptr);

// precondition: __ptr's reference count > 0
[[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI _LIBCPP_AVAILABILITY_SHARED_RESOURCE lock_guard<mutex>
__shared_resource_get_lock(const void* __ptr);

#  endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_NO_THREADS
#endif // _LIBCPP___UTILITY_SHARED_RESOURCE_H
