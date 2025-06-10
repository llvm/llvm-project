//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that setting _LIBCPP_ABI_BOUNDED_ITERATORS enabled bounded
// iterators in std::span and std::string_view, for historical reasons.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ABI_BOUNDED_ITERATORS

#include <version>

#ifndef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_SPAN
#  error _LIBCPP_ABI_BOUNDED_ITERATORS should enable bounded iterators in std::span
#endif

#ifndef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_STRING_VIEW
#  error _LIBCPP_ABI_BOUNDED_ITERATORS should enable bounded iterators in std::string_view
#endif
