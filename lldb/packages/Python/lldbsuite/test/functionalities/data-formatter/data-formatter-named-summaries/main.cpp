//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

struct First
{
    int x;
    int y;
    float dummy;
    First(int X, int Y) :
    x(X),
    y(Y),
    dummy(3.14)
    {}
};

struct Second
{
    int x;
    float y;
    Second(int X, float Y) :
    x(X),
    y(Y)
    {}
};

struct Third
{
    int x;
    char z;
    Third(int X, char Z) :
    x(X),
    z(Z)
    {}
};

int main (int argc, const char * argv[])
{
    First first(12,34);
    Second second(65,43.25);
    Third *third = new Third(96,'E');
    
    first.dummy = 1; // Set break point at this line.
    first.dummy = 2;
    first.dummy = 3;
    first.dummy = 4;
    first.dummy = 5;
    
}

