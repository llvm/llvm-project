//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <future>

// class future<R>

// future();

#include <future>
#include <cassert>

int main(int, char**)
{
    {
        std::future<int> f;
        assert(!f.valid());
    }
    {
        std::future<int&> f;
        assert(!f.valid());
    }
    {
        std::future<void> f;
        assert(!f.valid());
    }

  return 0;
}
