//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <memory>

// shared_ptr

// template<class T>
// bool
// atomic_is_lock_free(const shared_ptr<T>* p);

// UNSUPPORTED: c++98, c++03

#include <memory>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        const std::shared_ptr<int> p(new int(3));
        assert(std::atomic_is_lock_free(&p) == false);
    }

  return 0;
}
