//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX26_REMOVED_SHARED_PTR_ATOMICS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS
// UNSUPPORTED: no-threads

// <memory>

// shared_ptr

// template<class T>
// bool
// atomic_is_lock_free(const shared_ptr<T>* p);    // Deprecated in C++20, removed in C++26

// UNSUPPORTED: c++03

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
