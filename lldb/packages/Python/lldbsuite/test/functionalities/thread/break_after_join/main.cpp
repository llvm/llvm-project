//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which one thread will exit
// while a breakpoint is being handled in another thread.  This may not always
// happen because it's possible that the exiting thread will exit before the
// breakpoint is hit.  The test case should be flexible enough to treat that
// as success.

#include "pseudo_barrier.h"
#include <chrono>
#include <thread>

volatile int g_test = 0;

// A barrier to synchronize all the threads.
pseudo_barrier_t g_barrier1;

// A barrier to keep the threads from exiting until after the breakpoint has
// been passed.
pseudo_barrier_t g_barrier2;

void *
break_thread_func ()
{
    // Wait until all the threads are running
    pseudo_barrier_wait(g_barrier1);

    // Wait for the join thread to join
    std::this_thread::sleep_for(std::chrono::microseconds(50));

    // Do something
    g_test++;       // Set breakpoint here

    // Synchronize after the breakpoint
    pseudo_barrier_wait(g_barrier2);

    // Return
    return NULL;
}

void *
wait_thread_func ()
{
    // Wait until the entire first group of threads is running
    pseudo_barrier_wait(g_barrier1);

    // Wait until the breakpoint has been passed
    pseudo_barrier_wait(g_barrier2);

    // Return
    return NULL;
}

void *
join_thread_func (void *input)
{
    std::thread *thread_to_join = (std::thread *)input;

    // Sync up with the rest of the threads.
    pseudo_barrier_wait(g_barrier1);

    // Join the other thread
    thread_to_join->join();

    // Return
    return NULL;
}

int main ()
{
    // The first barrier waits for the non-joining threads to start.
    // This thread will also participate in that barrier.
    // The idea here is to guarantee that the joining thread will be
    // last in the internal list maintained by the debugger.
    pseudo_barrier_init(g_barrier1, 5);

    // The second barrier keeps the waiting threads around until the breakpoint
    // has been passed.
    pseudo_barrier_init(g_barrier2, 4);

    // Create a thread to hit the breakpoint
    std::thread thread_1(break_thread_func);

    // Create more threads to slow the debugger down during processing.
    std::thread thread_2(wait_thread_func);
    std::thread thread_3(wait_thread_func);
    std::thread thread_4(wait_thread_func);

    // Create a thread to join the breakpoint thread
    std::thread thread_5(join_thread_func, &thread_1);

    // Wait for the threads to finish
    thread_5.join();  // implies thread_1 is already finished
    thread_4.join();
    thread_3.join();
    thread_2.join();

    return 0;
}
