//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test makes sure that we use the correct implementation for comparing
// type_info objects on Apple platforms. See https://llvm.org/PR45549.

// REQUIRES: darwin

#include <typeinfo>

#if !defined(_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION)
#   error "_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION should be defined on Apple platforms"
#endif

#if defined(__x86_64__)
#   if _LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION != 1
#       error "_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION should be 1 (assume RTTI is merged) on Apple platforms"
#   endif
#elif defined(__aarch64__)
#   if _LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION != 3
#       error "_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION should be 3 (use the special ARM RTTI) on Apple platforms"
#   endif
#else
#   error "This test should be updated to pin down the RTTI behavior on this ABI."
#endif
