//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-clang-tidy

// RUN: %{clang-tidy} %s -header-filter=.* --checks='-*,libcpp-cpp-version-check' --load=%{test-tools-dir}/clang_tidy_checks/libcxx-tidy.plugin -- %{compile_flags} -fno-modules 2>&1 | %{check-output}

#include <__config>

// CHECK: warning: _LIBCPP_STD_VER >= version should be used instead of _LIBCPP_STD_VER > prev_version
#if _LIBCPP_STD_VER > 14
#endif

// CHECK: warning: Use _LIBCPP_STD_VER instead of __cplusplus to constrain based on the C++ version
#if __cplusplus >= 201103L
#endif

// CHECK: warning: _LIBCPP_STD_VER >= 11 is always true. Did you mean '#ifndef _LIBCPP_CXX03_LANG'?
#if _LIBCPP_STD_VER >= 11
#endif

// CHECK: warning: Not a valid value for _LIBCPP_STD_VER. Use 14, 17, 20, 23, or 26
#if _LIBCPP_STD_VER >= 12
#endif
