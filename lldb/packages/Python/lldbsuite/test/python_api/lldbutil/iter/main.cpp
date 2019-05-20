//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// C includes
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// C++ includes
#include <chrono>
#include <mutex>
#include <random>
#include <thread>

std::thread g_thread_1;
std::thread g_thread_2;
std::thread g_thread_3;
std::mutex g_mask_mutex;

typedef enum {
    eGet,
    eAssign,
    eClearBits
} MaskAction;

uint32_t mask_access (MaskAction action, uint32_t mask = 0);

uint32_t
mask_access (MaskAction action, uint32_t mask)
{
    static uint32_t g_mask = 0;

    std::lock_guard<std::mutex> lock(g_mask_mutex);
    switch (action)
    {
    case eGet:
        break;

    case eAssign:
        g_mask |= mask;
        break;

    case eClearBits:
        g_mask &= ~mask;
        break;
    }
    return g_mask;
}

void *
thread_func (void *arg)
{
    uint32_t thread_index = *((uint32_t *)arg);
    uint32_t thread_mask = (1u << (thread_index));
    printf ("%s (thread index = %u) startng...\n", __FUNCTION__, thread_index);

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, 3000000);

    while (mask_access(eGet) & thread_mask)
    {
        // random micro second sleep from zero to 3 seconds
        int usec = distribution(generator);
        printf ("%s (thread = %u) doing a usleep (%d)...\n", __FUNCTION__, thread_index, usec);

        std::chrono::microseconds duration(usec);
        std::this_thread::sleep_for(duration);
        printf ("%s (thread = %u) after usleep ...\n", __FUNCTION__, thread_index); // Set break point at this line.
    }
    printf ("%s (thread index = %u) exiting...\n", __FUNCTION__, thread_index);
    return NULL;
}


int main (int argc, char const *argv[])
{
    uint32_t thread_index_1 = 1;
    uint32_t thread_index_2 = 2;
    uint32_t thread_index_3 = 3;
    uint32_t thread_mask_1 = (1u << thread_index_1);
    uint32_t thread_mask_2 = (1u << thread_index_2);
    uint32_t thread_mask_3 = (1u << thread_index_3);

    // Make a mask that will keep all threads alive
    mask_access (eAssign, thread_mask_1 | thread_mask_2 | thread_mask_3); // And that line.

    // Create 3 threads
    g_thread_1 = std::thread(thread_func, (void*)&thread_index_1);
    g_thread_2 = std::thread(thread_func, (void*)&thread_index_2);
    g_thread_3 = std::thread(thread_func, (void*)&thread_index_3);

    char line[64];
    while (mask_access(eGet) != 0)
    {
        printf ("Enter thread index to kill or ENTER for all:\n");
        fflush (stdout);
        // Kill threads by index, or ENTER for all threads

        if (fgets (line, sizeof(line), stdin))
        {
            if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0')
            {
                printf ("Exiting all threads...\n");
                break;
            }
            int32_t index = strtoul (line, NULL, 0);
            switch (index)
            {
                case 1: mask_access (eClearBits, thread_mask_1); break;
                case 2: mask_access (eClearBits, thread_mask_2); break;
                case 3: mask_access (eClearBits, thread_mask_3); break;
            }
            continue;
        }

        break;
    }

    // Clear all thread bits to they all exit
    mask_access (eClearBits, UINT32_MAX);

    // Join all of our threads
    g_thread_1.join();
    g_thread_2.join();
    g_thread_3.join();

    return 0;
}
