//===-- safestack_util.h --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility code for SafeStack implementation.
//
//===----------------------------------------------------------------------===//

#ifndef SAFESTACK_UTIL_H
#define SAFESTACK_UTIL_H

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

namespace safestack {

#define SFS_CHECK(a)                                                  \
  do {                                                                \
    if (!(a)) {                                                       \
      fprintf(stderr, "safestack CHECK failed: %s:%d %s\n", __FILE__, \
              __LINE__, #a);                                          \
      abort();                                                        \
    };                                                                \
  } while (false)

inline size_t RoundUpTo(size_t size, size_t boundary) {
  SFS_CHECK((boundary & (boundary - 1)) == 0);
  return (size + boundary - 1) & ~(boundary - 1);
}

class MutexLock {
 public:
  explicit MutexLock(pthread_mutex_t &mutex) : mutex_(&mutex) {
    pthread_mutex_lock(mutex_);
  }
  ~MutexLock() { pthread_mutex_unlock(mutex_); }

 private:
  pthread_mutex_t *mutex_ = nullptr;
};

}  // namespace safestack

#endif  // SAFESTACK_UTIL_H
