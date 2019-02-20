//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <condition_variable>

// enum class cv_status { no_timeout, timeout };

#include <condition_variable>
#include <cassert>

int main(int, char**)
{
    assert(static_cast<int>(std::cv_status::no_timeout) == 0);
    assert(static_cast<int>(std::cv_status::timeout)    == 1);

  return 0;
}
