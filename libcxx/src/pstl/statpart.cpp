//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__pstl/backends/statpart.h>
#include <dispatch/dispatch.h>

// Hardcoded to use libdispatch for now.
// CMake switches to be introduced later.

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl::__statpart {

void __apply(size_t chunk_count, void* context, void (*func)(void* context, size_t chunk)) noexcept {
  ::dispatch_apply_f(chunk_count, DISPATCH_APPLY_AUTO, context, func);
}

} // namespace __pstl::__statpart
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
