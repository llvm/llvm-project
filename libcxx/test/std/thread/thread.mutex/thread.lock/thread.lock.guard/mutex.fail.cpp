//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads

// <mutex>

// template <class Mutex> class lock_guard;

// explicit lock_guard(mutex_type& m);

#include <mutex>

int main()
{
    std::mutex m;
    std::lock_guard<std::mutex> lg = m; // expected-error{{no viable conversion}}
}
