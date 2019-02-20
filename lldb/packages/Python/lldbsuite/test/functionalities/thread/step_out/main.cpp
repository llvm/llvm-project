//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which two threads are stopped
// at a breakpoint and the debugger issues a step-out command.

#include "pseudo_barrier.h"
#include <thread>

pseudo_barrier_t g_barrier;

volatile int g_test = 0;

void step_out_of_here() {
  g_test += 5; // Set breakpoint here
}

void *
thread_func ()
{
    // Wait until both threads are running
    pseudo_barrier_wait(g_barrier);

    // Do something
    step_out_of_here(); // Expect to stop here after step-out (clang)

    // Return
    return NULL;  // Expect to stop here after step-out (icc and gcc)
}

int main ()
{
    // Don't let either thread do anything until they're both ready.
    pseudo_barrier_init(g_barrier, 2);

    // Create two threads
    std::thread thread_1(thread_func);
    std::thread thread_2(thread_func);

    // Wait for the threads to finish
    thread_1.join();
    thread_2.join();

    return 0;
}
