//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
