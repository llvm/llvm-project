//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test is intended to create a situation in which one thread will exit
// while a breakpoint is being handled in another thread.  This may not always
// happen because it's possible that the exiting thread will exit before the
// breakpoint is hit.  The test case should be flexible enough to treat that
// as success.

#include <atomic>
#include <chrono>
#include <thread>

volatile int g_test = 0;

// A barrier to synchronize all the threads except the one that will exit.
pseudo_barrier_t g_barrier1;

// A barrier to synchronize all the threads including the one that will exit.
pseudo_barrier_t g_barrier2;

// A barrier to keep the first group of threads from exiting until after the
// breakpoint has been passed.
pseudo_barrier_t g_barrier3;

void *
break_thread_func ()
{
    // Wait until the entire first group of threads is running
    pseudo_barrier_wait(g_barrier1);

    // Wait for the exiting thread to start
    pseudo_barrier_wait(g_barrier2);

    // Do something
    g_test++;       // Set breakpoint here

    // Synchronize after the breakpoint
    pseudo_barrier_wait(g_barrier3);

    // Return
    return NULL;
}

void *
wait_thread_func ()
{
    // Wait until the entire first group of threads is running
    pseudo_barrier_wait(g_barrier1);

    // Wait for the exiting thread to start
    pseudo_barrier_wait(g_barrier2);

    // Wait until the breakpoint has been passed
    pseudo_barrier_wait(g_barrier3);

    // Return
    return NULL;
}

void *
exit_thread_func ()
{
    // Sync up with the rest of the threads.
    pseudo_barrier_wait(g_barrier2);

    // Try to make sure this thread doesn't exit until the breakpoint is hit.
    std::this_thread::sleep_for(std::chrono::microseconds(1));

    // Return
    return NULL;
}

int main ()
{

    // The first barrier waits for the non-exiting threads to start.
    // This thread will also participate in that barrier.
    // The idea here is to guarantee that the exiting thread will be
    // last in the internal list maintained by the debugger.
    pseudo_barrier_init(g_barrier1, 5);

    // The second break synchronyizes thread exection with the breakpoint.
    pseudo_barrier_init(g_barrier2, 5);

    // The third barrier keeps the waiting threads around until the breakpoint
    // has been passed.
    pseudo_barrier_init(g_barrier3, 4);

    // Create a thread to hit the breakpoint
    std::thread thread_1(break_thread_func);

    // Create more threads to slow the debugger down during processing.
    std::thread thread_2(wait_thread_func);
    std::thread thread_3(wait_thread_func);
    std::thread thread_4(wait_thread_func);

    // Wait for all these threads to get started.
    pseudo_barrier_wait(g_barrier1);

    // Create a thread to exit during the breakpoint
    std::thread thread_5(exit_thread_func);

    // Wait for the threads to finish
    thread_5.join();
    thread_4.join();
    thread_3.join();
    thread_2.join();
    thread_1.join();

    return 0;
}
