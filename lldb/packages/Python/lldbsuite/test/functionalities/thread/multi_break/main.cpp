//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which a breakpoint will be
// hit in two threads at nearly the same moment.  The expected result is that
// the breakpoint in the second thread will be hit while the breakpoint handler
// in the first thread is trying to stop all threads.

#include <atomic>
#include <thread>

pseudo_barrier_t g_barrier;

volatile int g_test = 0;

void *
thread_func ()
{
    // Wait until both threads are running
    pseudo_barrier_wait(g_barrier);

    // Do something
    g_test++;       // Set breakpoint here

    // Return
    return NULL;
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
