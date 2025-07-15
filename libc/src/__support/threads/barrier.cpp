//===-- Linux implementation of the callonce function ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/barrier.h"
#include "barrier.h"
#include "hdr/errno_macros.h"
#include "src/__support/threads/mutex.h"

namespace LIBC_NAMESPACE_DECL {

const int BARRIER_FIRST_EXITED = -1;

int Barrier::init(Barrier *b, const pthread_barrierattr_t *attr,
                  unsigned count) {
  LIBC_ASSERT(attr == nullptr); // TODO implement barrierattr
  if (count == 0)
    return EINVAL;

  b->expected = count;
  b->waiting = 0;
  b->blocking = true;

  int err;
  err = CndVar::init(&b->entering);
  if (err != 0)
    return err;

  err = CndVar::init(&b->exiting);
  if (err != 0)
    return err;

  Mutex::init(&b->m, false, false, false, false);
  return 0;
}

int Barrier::wait() {
  m.lock();

  // if the barrier is emptying out threads, wait until it finishes
  while (!blocking) {
    entering.wait(&m);
  }
  waiting++;

  if (waiting == expected) {
    // this is the last thread to call wait(), so lets wake everyone up
    blocking = false;
    exiting.broadcast();
  } else {
    // block threads until waiting = expected
    while (blocking) {
      exiting.wait(&m);
    }
  }
  waiting--;

  // all threads have exited the barrier, lets let the ones waiting to enter
  // continue
  if (waiting == 0) {
    blocking = true;
    entering.broadcast();
    m.unlock();
    return BARRIER_FIRST_EXITED;
  }
  m.unlock();

  return 0;
}

int Barrier::destroy(Barrier *b) {
  CndVar::destroy(&b->entering);
  CndVar::destroy(&b->exiting);
  Mutex::destroy(&b->m);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
