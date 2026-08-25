//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__assert>
#include <new>

#if !defined(__GLIBCXX__) && !defined(_LIBCPP_ABI_VCRUNTIME)

inline void __throw_bad_alloc_shim() { std::__throw_bad_alloc(); }

#  define _LIBCPP_ASSERT_SHIM(expr, str) _LIBCPP_ASSERT(expr, str)

#  include "support/new.ipp"

#endif // !__GLIBCXX__ && !_LIBCPP_ABI_VCRUNTIME
