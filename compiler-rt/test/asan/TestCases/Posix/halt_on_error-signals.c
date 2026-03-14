// Test interaction of Asan recovery mode with asynch signals.
//
// RUN: %clang_asan -fsanitize-recover=address -pthread %s -o %t
//
// RUN: %env_asan_opts=halt_on_error=false:suppress_equal_pcs=false %run %t 100 >%t.log 2>&1 || true
// Collision will almost always get triggered but we still need to check the unlikely case:
// RUN: FileCheck --check-prefix=CHECK-COLLISION %s <%t.log || FileCheck --check-prefix=CHECK-NO-COLLISION %s <%t.log

#define _SVID_SOURCE 1  // SA_NODEFER

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>

#include <sanitizer/asan_interface.h>

void random_delay(unsigned *seed) {
  *seed = 1664525 * *seed + 1013904223;
  struct timespec delay = { 0, (*seed % 1000) * 1000 };
  nanosleep(&delay, 0);
}

volatile char bad[2] = {1, };

void error() {
  // CHECK-COLLISION: AddressSanitizer: nested bug in the same thread, aborting
  // CHECK-NO-COLLISION: AddressSanitizer: use-after-poison
  volatile int idx = 0;
  bad[idx] = 0;
}

#define CHECK_CALL(e, msg) do {             \
  if (0 != (e)) {                           \
    fprintf(stderr, "Failed to " msg "\n"); \
    exit(1);                                \
  }                                         \
} while (0)

size_t niter = 10;
pthread_t sender_tid, receiver_tid;

pthread_mutex_t keep_alive_mu = PTHREAD_MUTEX_INITIALIZER;

void *sender(void *arg) {
  unsigned seed = 0;
  for (size_t i = 0; i < niter; ++i) {
    random_delay(&seed);
    CHECK_CALL(pthread_kill(receiver_tid, SIGUSR1), "send signal");
  }
  return 0;
}

void handler(int sig) {
  // Expect error collisions here
  error();
}

void *receiver(void *arg) {
  unsigned seed = 1;
  for (size_t i = 0; i < niter; ++i) {
    random_delay(&seed);
    // And here
    error();
  }
  // Parent will release this when it's ok to terminate
  CHECK_CALL(pthread_mutex_lock(&keep_alive_mu), "unlock mutex");
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Syntax: %s niter\n", argv[0]);
    exit(1);
  }

  niter = (size_t)strtoul(argv[1], 0, 0);

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = handler;
  sa.sa_flags = SA_NODEFER; // Enable nested handlers to add more stress
  CHECK_CALL(sigaction(SIGUSR1, &sa, 0), "set sighandler");

  __asan_poison_memory_region(&bad, sizeof(bad));

  CHECK_CALL(pthread_mutex_lock(&keep_alive_mu), "lock mutex");
  CHECK_CALL(pthread_create(&receiver_tid, 0, receiver, 0), "start thread");
  CHECK_CALL(pthread_create(&sender_tid, 0, sender, 0), "start thread");
  CHECK_CALL(pthread_join(sender_tid, 0), "join thread");
  // Now allow receiver to die
  CHECK_CALL(pthread_mutex_unlock(&keep_alive_mu), "unlock mutex");
  CHECK_CALL(pthread_join(receiver_tid, 0), "join thread");

  // CHECK-NO-COLLISION: All threads terminated
  printf("All threads terminated\n");

  return 0;
}
