// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s

// Test case for recursive signal handlers, adopted from:
// https://github.com/google/sanitizers/issues/478

// REQUIRES: disabled

#include "test.h"
#include <semaphore.h>
#include <signal.h>
#include <errno.h>

static const int kSigSuspend = SIGUSR1;
static const int kSigRestart = SIGUSR2;

static sem_t g_thread_suspend_ack_sem;

static bool g_busy_thread_received_restart;

static volatile bool g_busy_thread_garbage_collected;

static void SaveRegistersInStack() {
  // Mono walks thread stacks to detect unreferenced objects.
  // If last object reference is kept in register the object will be collected
  // This is why threads can't be suspended with something like pthread_suspend
}

static void fail(const char *what) {
  fprintf(stderr, "FAILED: %s (errno=%d)\n", what, errno);
  exit(1);
}

static void SuspendHandler(int sig) {
  int old_errno = errno;
  SaveRegistersInStack();

  // Enable kSigRestart handling, tsan disables signals around signal handlers.
  sigset_t sigset;
  sigemptyset(&sigset);
  pthread_sigmask(SIG_SETMASK, &sigset, 0);

  // Acknowledge that thread is saved and suspended
  if (sem_post(&g_thread_suspend_ack_sem) != 0)
    fail("sem_post failed");

  // Wait for wakeup signal.
  while (!g_busy_thread_received_restart)
    usleep(100);  // wait for kSigRestart signal

  // Acknowledge that thread restarted
  if (sem_post(&g_thread_suspend_ack_sem) != 0)
    fail("sem_post failed");

  g_busy_thread_garbage_collected = true;

  errno = old_errno;
}

static void RestartHandler(int sig) {
  g_busy_thread_received_restart = true;
}

static void StopWorld(pthread_t thread) {
  if (pthread_kill(thread, kSigSuspend) != 0)
    fail("pthread_kill failed");

  while (sem_wait(&g_thread_suspend_ack_sem) != 0) {
    if (errno != EINTR)
      fail("sem_wait failed");
  }
}

static void StartWorld(pthread_t thread) {
  if (pthread_kill(thread, kSigRestart) != 0)
    fail("pthread_kill failed");

  while (sem_wait(&g_thread_suspend_ack_sem) != 0) {
    if (errno != EINTR)
      fail("sem_wait failed");
  }
}

static void CollectGarbage(pthread_t thread) {
  StopWorld(thread);
  // Walk stacks
  StartWorld(thread);
}

static void Init() {
  if (sem_init(&g_thread_suspend_ack_sem, 0, 0) != 0)
    fail("sem_init failed");

  struct sigaction act = {};
  act.sa_flags = SA_RESTART;
  act.sa_handler = &SuspendHandler;
  if (sigaction(kSigSuspend, &act, NULL) != 0)
    fail("sigaction failed");
  act.sa_handler = &RestartHandler;
  if (sigaction(kSigRestart, &act, NULL) != 0)
    fail("sigaction failed");
}

void* BusyThread(void *arg) {
  (void)arg;
  while (!g_busy_thread_garbage_collected) {
    usleep(100); // Tsan deadlocks without these sleeps
  }
  return NULL;
}

int main(int argc, const char *argv[]) {
  Init();
  pthread_t busy_thread;
  if (pthread_create(&busy_thread, NULL, &BusyThread, NULL) != 0)
    fail("pthread_create failed");
  CollectGarbage(busy_thread);
  if (pthread_join(busy_thread, 0) != 0)
    fail("pthread_join failed");
  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK-NOT: FAILED
// CHECK-NOT: ThreadSanitizer CHECK failed
// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: DONE
