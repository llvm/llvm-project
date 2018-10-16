//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pseudo_barrier.h"
#include <cstdio>
#include <thread>

volatile uint32_t g_val = 0;
pseudo_barrier_t g_barrier;

void thread_func() {
  pseudo_barrier_wait(g_barrier);
  printf("%s starting...\n", __FUNCTION__);
  for (uint32_t i = 0; i < 10; ++i)
    g_val = i;
}

int main(int argc, char const *argv[]) {
  printf("Before running the thread\n");
  pseudo_barrier_init(g_barrier, 2);
  std::thread thread(thread_func);

  printf("After running the thread\n");
  pseudo_barrier_wait(g_barrier);

  thread.join();

  return 0;
}
