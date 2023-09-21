//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// OpenMP target offloading has only been supported since version 4.5. This test
// verifies that one can include algorithm without any diagnostics when using a
// version that is newer than the minimum requirement.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: -fopenmp -fopenmp-version=51

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>

// expected-no-diagnostics
