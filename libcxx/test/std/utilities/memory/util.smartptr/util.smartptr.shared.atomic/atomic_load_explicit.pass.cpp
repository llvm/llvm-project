//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
//
// This test uses new symbols that were not defined in the libc++ shipped on
// darwin11 and darwin12:
// XFAIL: availability=macosx10.7
// XFAIL: availability=macosx10.8

// <memory>

// shared_ptr

// template <class T>
// shared_ptr<T>
// atomic_load_explicit(const shared_ptr<T>* p, memory_order mo)

// UNSUPPORTED: c++98, c++03

#include <memory>
#include <cassert>

#include "test_macros.h"

int main()
{
    {
        const std::shared_ptr<int> p(new int(3));
        std::shared_ptr<int> q = std::atomic_load_explicit(&p, std::memory_order_relaxed);
        assert(*q == *p);
    }
}
