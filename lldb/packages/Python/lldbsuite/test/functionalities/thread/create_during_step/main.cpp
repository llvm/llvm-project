//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which one thread will be
// created while the debugger is stepping in another thread.

#include "pseudo_barrier.h"
#include <thread>

#define do_nothing()

pseudo_barrier_t g_barrier;

volatile int g_thread_created = 0;
volatile int g_test = 0;

void *
step_thread_func ()
{
    g_test = 0;         // Set breakpoint here

    while (!g_thread_created)
        g_test++;

    // One more time to provide a continue point
    g_test++;           // Continue from here

    // Return
    return NULL;
}

void *
create_thread_func (void *input)
{
    std::thread *step_thread = (std::thread*)input;

    // Wait until the main thread knows this thread is started.
    pseudo_barrier_wait(g_barrier);

    // Wait until the other thread is done.
    step_thread->join();

    // Return
    return NULL;
}

int main ()
{
    // Use a simple count to simulate a barrier.
    pseudo_barrier_init(g_barrier, 2);

    // Create a thread to hit the breakpoint.
    std::thread thread_1(step_thread_func);

    // Wait until the step thread is stepping
    while (g_test < 1)
        do_nothing();

    // Create a thread to exit while we're stepping.
    std::thread thread_2(create_thread_func, &thread_1);

    // Wait until that thread is started
    pseudo_barrier_wait(g_barrier);

    // Let the stepping thread know the other thread is there
    g_thread_created = 1;

    // Wait for the threads to finish.
    thread_2.join();
    thread_1.join();

    return 0;
}
