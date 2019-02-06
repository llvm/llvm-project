//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// function(nullptr_t);

#include <functional>
#include <cassert>

int main()
{
    std::function<int(int)> f(nullptr);
    assert(!f);
}
