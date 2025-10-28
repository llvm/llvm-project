//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure the following libcxx override can coexists with standard C function.
// std::__ibm int nanosleep(const struct timespec* , struct timespec* );

#include <time.h> // timespec

int nanosleep(const struct timespec*, struct timespec*);
#include <mutex>
