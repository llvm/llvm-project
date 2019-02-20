//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

bool bar(int const *foo) {
  return foo != 0; // Set break point at this line.
}

int main() {
  int foo[] = {1,2,3};
  return bar(foo);
}
