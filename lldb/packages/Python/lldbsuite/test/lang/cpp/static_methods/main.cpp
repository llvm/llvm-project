//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

class A
{
public:
  static int getStaticValue();
  int getMemberValue();
  int a;
};

int A::getStaticValue()
{
  return 5;
} 

int A::getMemberValue()
{
  return a;
}

int main()
{
  A my_a;

  my_a.a = 3;

  printf("%d\n", A::getStaticValue()); // Break at this line
  printf("%d\n", my_a.getMemberValue());
}
