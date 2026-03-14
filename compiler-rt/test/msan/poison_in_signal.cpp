// Stress test of poisoning from signal handler.

// RUN: %clangxx_msan -std=c++11 -O2 %s -o %t && %run %t
// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -std=c++11 -O2 %s -o %t && %run %t
// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -fsanitize-memory-use-after-dtor -std=c++11 -O2 %s -o %t && %run %t

#include <assert.h>
#include <atomic>
#include <pthread.h>
#include <signal.h>
#include <sys/time.h>

#include <sanitizer/msan_interface.h>

std::atomic<int> n = {1000};

struct Tmp {
  char buff[1];
  ~Tmp() {}
};

__attribute__((noinline, optnone)) void Poison() {
  // use-after-dtor.
  volatile Tmp t;
  // Regular poisoning.
  __msan_poison(&t, sizeof(t));
}

void *thr(void *p) {
  for (; n >= 0;) {
    for (int i = 0; i < 1000; i++) {
      Poison();
    }
  }
  return 0;
}

void handler(int) {
  Poison();
  --n;
}

int main(int argc, char **argv) {
  const int kThreads = 10;
  pthread_t th[kThreads];
  for (int i = 0; i < kThreads; i++)
    pthread_create(&th[i], 0, thr, 0);

  struct sigaction sa = {};
  sa.sa_handler = handler;
  assert(!sigaction(SIGPROF, &sa, 0));

  itimerval t;
  t.it_value.tv_sec = 0;
  t.it_value.tv_usec = 10;
  t.it_interval = t.it_value;
  assert(!setitimer(ITIMER_PROF, &t, 0));

  for (int i = 0; i < kThreads; i++)
    pthread_join(th[i], 0);

  return 0;
}
