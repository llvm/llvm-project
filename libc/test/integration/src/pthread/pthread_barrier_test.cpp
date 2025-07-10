//===-- Tests for pthread_barrier_t ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_barrier_destroy.h"
#include "src/pthread/pthread_barrier_init.h"
#include "src/pthread/pthread_barrier_wait.h"

#include "src/pthread/pthread_create.h"

#include "test/IntegrationTest/test.h"

#include <pthread.h>
#include <stdint.h> // uintptr_t

constexpr int START = 0;
constexpr int MAX = 10000;

pthread_barrier_t barrier;
static int shared_int = START;

void increment_shared_counter() {

}



TEST_MAIN() {
  
  return 0;
}
