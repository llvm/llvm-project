// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <vector>
#include <algorithm>
#include <sys/time.h>

const int kThreads = 4;
const int kMutexes = 16 << 10;
const int kIters = 400 << 10;
const int kMaxPerThread = 10;

const int kStateInited = 0;
const int kStateNotInited = -1;
const int kStateLocked = -2;

struct Mutex {
  int state;
  pthread_rwlock_t m;
};

Mutex mtx[kMutexes];

void check(int res) {
  if (res != 0) {
    printf("SOMETHING HAS FAILED\n");
    exit(1);
  }
}

bool cas(int *a, int oldval, int newval) {
  return __atomic_compare_exchange_n(a, &oldval, newval, false,
      __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
}

void *Thread(void *seed) {
  unsigned rnd = (unsigned)(unsigned long)seed;
  int err;
  std::vector<int> locked;
  for (int i = 0; i < kIters; i++) {
    int what = rand_r(&rnd) % 10;
    if (what < 4 && locked.size() < kMaxPerThread) {
      // lock
      int max_locked = -1;
      if (!locked.empty()) {
        max_locked = *std::max_element(locked.begin(), locked.end());
        if (max_locked == kMutexes - 1) {
          i--;
          continue;
        }
      }
      int id = (rand_r(&rnd) % (kMutexes - max_locked - 1)) + max_locked + 1;
      Mutex *m = &mtx[id];
      // init the mutex if necessary or acquire a reference
      for (;;) {
        int old = __atomic_load_n(&m->state, __ATOMIC_RELAXED);
        if (old == kStateLocked) {
          sched_yield();
          continue;
        }
        int newv = old + 1;
        if (old == kStateNotInited)
          newv = kStateLocked;
        if (cas(&m->state, old, newv)) {
          if (old == kStateNotInited) {
            if ((err = pthread_rwlock_init(&m->m, 0))) {
              fprintf(stderr, "pthread_rwlock_init failed with %d\n", err);
              exit(1);
            }
            if (!cas(&m->state, kStateLocked, 1)) {
              fprintf(stderr, "init commit failed\n");
              exit(1);
            }
          }
          break;
        }
      }
      // now we have an inited and referenced mutex, choose what to do
      bool failed = false;
      switch (rand_r(&rnd) % 4) {
      case 0:
        if ((err = pthread_rwlock_wrlock(&m->m))) {
          fprintf(stderr, "pthread_rwlock_wrlock failed with %d\n", err);
          exit(1);
        }
        break;
      case 1:
        if ((err = pthread_rwlock_rdlock(&m->m))) {
          fprintf(stderr, "pthread_rwlock_rdlock failed with %d\n", err);
          exit(1);
        }
        break;
      case 2:
        err = pthread_rwlock_trywrlock(&m->m);
        if (err != 0 && err != EBUSY) {
          fprintf(stderr, "pthread_rwlock_trywrlock failed with %d\n", err);
          exit(1);
        }
        failed = err == EBUSY;
        break;
      case 3:
        err = pthread_rwlock_tryrdlock(&m->m);
        if (err != 0 && err != EBUSY) {
          fprintf(stderr, "pthread_rwlock_tryrdlock failed with %d\n", err);
          exit(1);
        }
        failed = err == EBUSY;
        break;
      }
      if (failed) {
        if (__atomic_fetch_sub(&m->state, 1, __ATOMIC_ACQ_REL) <= 0) {
          fprintf(stderr, "failed to unref after failed trylock\n");
          exit(1);
        }
        continue;
      }
      locked.push_back(id);
    } else if (what < 9 && !locked.empty()) {
      // unlock
      int pos = rand_r(&rnd) % locked.size();
      int id = locked[pos];
      locked[pos] = locked[locked.size() - 1];
      locked.pop_back();
      Mutex *m = &mtx[id];
      if ((err = pthread_rwlock_unlock(&m->m))) {
        fprintf(stderr, "pthread_rwlock_unlock failed with %d\n", err);
        exit(1);
      }
      if (__atomic_fetch_sub(&m->state, 1, __ATOMIC_ACQ_REL) <= 0) {
        fprintf(stderr, "failed to unref after unlock\n");
        exit(1);
      }
    } else {
      // Destroy a random mutex.
      int id = rand_r(&rnd) % kMutexes;
      Mutex *m = &mtx[id];
      if (!cas(&m->state, kStateInited, kStateLocked)) {
        i--;
        continue;
      }
      if ((err = pthread_rwlock_destroy(&m->m))) {
        fprintf(stderr, "pthread_rwlock_destroy failed with %d\n", err);
        exit(1);
      }
      if (!cas(&m->state, kStateLocked, kStateNotInited)) {
        fprintf(stderr, "destroy commit failed\n");
        exit(1);
      }
    }
  }
  // Unlock all previously locked mutexes, otherwise other threads can deadlock.
  for (int i = 0; i < locked.size(); i++) {
    int id = locked[i];
    Mutex *m = &mtx[id];
    if ((err = pthread_rwlock_unlock(&m->m))) {
      fprintf(stderr, "pthread_rwlock_unlock failed with %d\n", err);
      exit(1);
    }
  }
  return 0;
}

int main() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  unsigned s = tv.tv_sec + tv.tv_usec;
  fprintf(stderr, "seed %d\n", s);
  srand(s);
  for (int i = 0; i < kMutexes; i++)
    mtx[i].state = kStateNotInited;
  pthread_t t[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&t[i], 0, Thread, (void*)(unsigned long)rand());
  for (int i = 0; i < kThreads; i++)
    pthread_join(t[i], 0);
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer
// CHECK: DONE

