// This test checks that the implementation of use-after-return
// is async-signal-safe.
// RUN: %clangxx_asan -std=c++11 -O1 %s -o %t -pthread && %run %t
// REQUIRES: stable-runtime
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <pthread.h>
#include <initializer_list>

int *g;
int n_signals;

typedef void (*Sigaction)(int, siginfo_t *, void *);

void SignalHandler(int, siginfo_t*, void*) {
  int local;
  g = &local;
  n_signals++;
}

static void EnableSigprof(Sigaction SignalHandler) {
  struct sigaction sa;
  sa.sa_sigaction = SignalHandler;
  sa.sa_flags = SA_RESTART | SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGPROF, &sa, NULL) != 0) {
    perror("sigaction");
    abort();
  }
  struct itimerval timer;
  timer.it_interval.tv_sec = 0;
  timer.it_interval.tv_usec = 1;
  timer.it_value = timer.it_interval;
  if (setitimer(ITIMER_PROF, &timer, 0) != 0) {
    perror("setitimer");
    abort();
  }
}

void RecursiveFunction(int depth) {
  if (depth == 0) return;
  int local;
  g = &local;
  // printf("r: %p\n", &local);
  // printf("[%2d] n_signals: %d\n", depth, n_signals);
  RecursiveFunction(depth - 1);
  RecursiveFunction(depth - 1);
}

void *FastThread(void *) {
  RecursiveFunction(1);
  return NULL;
}

void *SlowThread(void *) {
  RecursiveFunction(1);
  return NULL;
}

int main(int argc, char **argv) {
  EnableSigprof(SignalHandler);

  for (auto Thread : {&FastThread, &SlowThread}) {
    for (int i = 0; i < 1000; i++) {
      fprintf(stderr, ".");
      const int kNumThread = sizeof(void*) == 8 ? 32 : 8;
      pthread_t t[kNumThread];
      for (int i = 0; i < kNumThread; i++)
        pthread_create(&t[i], 0, Thread, 0);
      for (int i = 0; i < kNumThread; i++)
        pthread_join(t[i], 0);
    }
    fprintf(stderr, "\n");
  }
}
