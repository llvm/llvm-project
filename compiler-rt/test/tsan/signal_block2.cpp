// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// The test was reported to hang sometimes on Darwin:
// https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20210517/917003.html
// UNSUPPORTED: darwin

#include "test.h"
#include <signal.h>
#include <string.h>
#include <sys/time.h>

int test;
int done;
int signals_handled;
pthread_t main_thread;
pthread_mutex_t mutex;
pthread_cond_t cond;

void timer_handler(int signum) {
  write(2, "timer_handler\n", strlen("timer_handler\n"));
  if (++signals_handled < 10)
    return;
  switch (test) {
  case 0:
    __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
    (void)pthread_kill(main_thread, SIGUSR1);
  case 1:
    if (pthread_mutex_trylock(&mutex) == 0) {
      __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
      pthread_cond_signal(&cond);
      pthread_mutex_unlock(&mutex);
    }
  case 2:
    __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
  }
}

int main(int argc, char **argv) {
  main_thread = pthread_self();
  pthread_mutex_init(&mutex, 0);
  pthread_cond_init(&cond, 0);

  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGUSR1);
  if (sigprocmask(SIG_BLOCK, &sigset, NULL))
    exit((perror("sigprocmask"), 1));

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = &timer_handler;
  if (sigaction(SIGALRM, &sa, NULL))
    exit((perror("setitimer"), 1));

  for (test = 0; test < 3; test++) {
    fprintf(stderr, "test %d\n", test);
    struct itimerval timer;
    timer.it_value.tv_sec = 0;
    timer.it_value.tv_usec = 50000;
    timer.it_interval = timer.it_value;
    if (setitimer(ITIMER_REAL, &timer, NULL))
      exit((perror("setitimer"), 1));

    switch (test) {
    case 0:
      while (__atomic_load_n(&done, __ATOMIC_ACQUIRE) == 0) {
        int signum;
        sigwait(&sigset, &signum);
        write(2, "sigwait\n", strlen("sigwait\n"));
      }
    case 1:
      pthread_mutex_lock(&mutex);
      while (__atomic_load_n(&done, __ATOMIC_ACQUIRE) == 0) {
        pthread_cond_wait(&cond, &mutex);
        write(2, "pthread_cond_wait\n", strlen("pthread_cond_wait\n"));
      }
      pthread_mutex_unlock(&mutex);
    case 2:
      while (__atomic_load_n(&done, __ATOMIC_ACQUIRE) == 0) {
      }
    }

    memset(&timer, 0, sizeof(timer));
    if (setitimer(ITIMER_REAL, &timer, NULL))
      exit((perror("setitimer"), 1));
    done = 0;
    signals_handled = 0;
  }
  fprintf(stderr, "DONE\n");
}

// CHECK: DONE
