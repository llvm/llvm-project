//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>

std::default_random_engine g_random_engine{std::random_device{}()};
std::uniform_int_distribution<> g_distribution{0, 3000000};

uint32_t g_val = 0;


uint32_t
access_pool (bool flag = false)
{
    static std::mutex g_access_mutex;
    g_access_mutex.lock();

    uint32_t old_val = g_val;
    if (flag)
    {
        printf("changing g_val to %d...\n", old_val + 1);
        g_val = old_val + 1;
    }

    g_access_mutex.unlock();
    return g_val;
}

void
thread_func (uint32_t thread_index)
{
    printf ("%s (thread index = %u) starting...\n", __FUNCTION__, thread_index);

    uint32_t count = 0;
    uint32_t val;
    while (count++ < 15)
    {
        // random micro second sleep from zero to 3 seconds
        int usec = g_distribution(g_random_engine);
        printf ("%s (thread = %u) doing a usleep (%d)...\n", __FUNCTION__, thread_index, usec);
        std::this_thread::sleep_for(std::chrono::microseconds{usec});

        if (count < 7)
            val = access_pool ();
        else
            val = access_pool (true);

        printf ("%s (thread = %u) after usleep access_pool returns %d (count=%d)...\n", __FUNCTION__, thread_index, val, count);
    }
    printf ("%s (thread index = %u) exiting...\n", __FUNCTION__, thread_index);
}


int main (int argc, char const *argv[])
{
    std::thread threads[3];

    printf ("Before turning all three threads loose...\n"); // Set break point at this line,
                                                            // in order to set our watchpoint.
    // Create 3 threads
    for (auto &thread : threads)
        thread = std::thread{thread_func, std::distance(threads, &thread)};

    // Join all of our threads
    for (auto &thread : threads)
        thread.join();

    return 0;
}
