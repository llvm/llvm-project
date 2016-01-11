//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

struct S {
  int x;
  int y;
  
  S() : x(123), y(456) {}
};

int main() {
  S object;
  return 0; // break here
}
