//===-- main.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
