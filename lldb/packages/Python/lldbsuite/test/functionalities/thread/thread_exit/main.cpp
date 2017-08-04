//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This test verifies the correct handling of child thread exits.

#include <atomic>
#include <thread>

pseudo_barrier_t g_barrier1;
pseudo_barrier_t g_barrier2;
pseudo_barrier_t g_barrier3;

void *
thread1 ()
{
    // Synchronize with the main thread.
    pseudo_barrier_wait(g_barrier1);

    // Synchronize with the main thread and thread2.
    pseudo_barrier_wait(g_barrier2);

    // Return
    return NULL;                                      // Set second breakpoint here
}

void *
thread2 ()
{
    // Synchronize with thread1 and the main thread.
    pseudo_barrier_wait(g_barrier2);

    // Synchronize with the main thread.
    pseudo_barrier_wait(g_barrier3);

    // Return
    return NULL;
}

int main ()
{
    pseudo_barrier_init(g_barrier1, 2);
    pseudo_barrier_init(g_barrier2, 3);
    pseudo_barrier_init(g_barrier3, 2);

    // Create a thread.
    std::thread thread_1(thread1);

    // Wait for thread1 to start.
    pseudo_barrier_wait(g_barrier1);

    // Create another thread.
    std::thread thread_2(thread2);  // Set first breakpoint here

    // Wait for thread2 to start.
    pseudo_barrier_wait(g_barrier2);

    // Wait for the first thread to finish
    thread_1.join();

    // Synchronize with the remaining thread
    pseudo_barrier_wait(g_barrier3);                  // Set third breakpoint here

    // Wait for the second thread to finish
    thread_2.join();

    return 0;                                         // Set fourth breakpoint here
}
