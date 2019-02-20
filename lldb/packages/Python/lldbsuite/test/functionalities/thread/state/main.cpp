//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is intended to verify that thread states are properly maintained
// when transitional actions are performed in the debugger.  Most of the logic
// is in the test script.  This program merely provides places where the test
// can create the intended states.

#include <chrono>
#include <thread>

volatile int g_test = 0;

int addSomething(int a)
{
    return a + g_test;
}

int doNothing()
{
    int temp = 0;   // Set first breakpoint here

    while (!g_test && temp < 5)
    {
        ++temp;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    return temp;    // Set second breakpoint here
}

int main ()
{
    int result = doNothing();

    int i = addSomething(result);

    return 0;
}
