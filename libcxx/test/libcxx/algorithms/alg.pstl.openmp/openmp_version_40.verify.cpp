//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// OpenMP target offloading has only been supported since version 4.5. This test
// verifies that a diagnostic error is prompted if the OpenMP version is below
// the minimum required version.

// UNSUPPORTED: c++03, c++11, c++14, gcc

// ADDITIONAL_COMPILE_FLAGS: -fopenmp -fopenmp-version=40

// REQUIRES: libcpp-pstl-backend-openmp

#include <algorithm>

// expected-error@__algorithm/pstl_backends/openmp/backend.h:26 {{"OpenMP target offloading has been supported since OpenMP version 4.5 (201511). Please use a more recent version of OpenMP."}}
