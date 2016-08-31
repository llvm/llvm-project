//===-- main.cpp -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

struct Foo
{
  double x;
  int y;
  Foo() : x(3.1415), y(1234) {}
};

int main() {
  Foo f;
  return 0; // break here
}
