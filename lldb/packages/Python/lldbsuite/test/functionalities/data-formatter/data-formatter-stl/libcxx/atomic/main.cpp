//===-- main.cpp --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

