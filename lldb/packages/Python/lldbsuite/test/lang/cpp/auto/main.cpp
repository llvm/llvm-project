//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>

int main()
{
  std::string helloworld("hello world");

  // Ensure std::string copy constructor is present in the binary, as we will
  // use it in an expression.
  std::string other = helloworld;

  return 0; // break here
}
