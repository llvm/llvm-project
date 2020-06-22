//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// <future>

// class packaged_task<R(ArgTypes...)>

// packaged_task& operator=(packaged_task&) = delete;

#include <future>

int main(int, char**)
{
    {
        std::packaged_task<double(int, char)> p0, p;
        p = p0; // expected-error {{overload resolution selected deleted operator '='}}
    }

  return 0;
}
