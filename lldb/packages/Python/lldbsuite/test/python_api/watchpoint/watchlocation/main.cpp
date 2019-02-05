//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <random>
#include <thread>

std::default_random_engine g_random_engine{std::random_device{}()};
std::uniform_int_distribution<> g_distribution{0, 3000000};
std::condition_variable g_condition_variable;
std::mutex g_mutex;
int g_count;

char *g_char_ptr = nullptr;

void
barrier_wait()
{
    std::unique_lock<std::mutex> lock{g_mutex};
    if (--g_count > 0)
        g_condition_variable.wait(lock);
    else
        g_condition_variable.notify_all();
}

void
do_bad_thing_with_location(char *char_ptr, char new_val)
{
    *char_ptr = new_val;
}

uint32_t
access_pool (bool flag = false)
{
    static std::mutex g_access_mutex;
    if (!flag)
        g_access_mutex.lock();

    char old_val = *g_char_ptr;
    if (flag)
        do_bad_thing_with_location(g_char_ptr, old_val + 1);

    if (!flag)
        g_access_mutex.unlock();
    return *g_char_ptr;
}

void
thread_func (uint32_t thread_index)
{
    printf ("%s (thread index = %u) startng...\n", __FUNCTION__, thread_index);

    barrier_wait();

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
    g_count = 4;
    std::thread threads[3];

    g_char_ptr = new char{};

    // Create 3 threads
    for (auto &thread : threads)
        thread = std::thread{thread_func, std::distance(threads, &thread)};

    printf ("Before turning all three threads loose...\n"); // Set break point at this line.
    barrier_wait();

    // Join all of our threads
    for (auto &thread : threads)
        thread.join();

    delete g_char_ptr;

    return 0;
}
