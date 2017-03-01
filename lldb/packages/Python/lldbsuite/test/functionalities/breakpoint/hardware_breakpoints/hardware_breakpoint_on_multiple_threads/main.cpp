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

#define NUM_OF_THREADS 4

std::mutex hw_break_mutex;

void
hw_break_function (uint32_t thread_index) {
  printf ("%s called by Thread #%u...\n", __FUNCTION__, thread_index);
}


void
thread_func (uint32_t thread_index) {
  printf ("%s (thread index = %u) starting...\n", __FUNCTION__, thread_index);

  hw_break_mutex.lock();
  
  hw_break_function(thread_index); // Call hw_break_function

  hw_break_mutex.unlock();
}


int main (int argc, char const *argv[])
{
  std::thread threads[NUM_OF_THREADS]; 

  printf ("Starting thread creation with hardware breakpoint set...\n");

  for (auto &thread : threads)
    thread = std::thread{thread_func, std::distance(threads, &thread)};

  for (auto &thread : threads)
    thread.join();

  return 0;
}
