//===-- main.cpp --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>

struct S {
    int x = 1;
    int y = 2;
};

int main ()
{
    std::atomic<S> s;
    s.store(S());
    std::atomic<int> i;
    i.store(5);
    
    return 0; // Set break point at this line.
}

