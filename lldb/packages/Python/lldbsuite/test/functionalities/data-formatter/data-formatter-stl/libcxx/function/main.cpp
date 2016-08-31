//===-- main.cpp --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <functional>

int foo(int x, int y) {
  return x + y - 1;
}

int main ()
{
  int acc = 42;
  std::function<int (int,int)> f1 = foo;
  std::function<int (int)> f2 = [acc,f1] (int x) -> int {
    return x+f1(acc,x);
  };
    return f1(acc,acc) + f2(acc); // Set break point at this line.
}

