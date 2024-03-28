// RUN: %clang_tsan %s -lstdc++ -o %t && %run %t 2>&1 | FileCheck %s

#include "../test.h"
#include <errno.h>
#include <linux/futex.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <sys/syscall.h>

#include <cassert>
#include <stdexcept>
#include <thread>

#include <sanitizer/linux_syscall_hooks.h>

int futex(int *uaddr, int futex_op, int val, const struct timespec *timeout,
          int *uaddr2, int val3) {
  __sanitizer_syscall_pre_futex(uaddr, futex_op, val, timeout, uaddr2, val3);
  int result = syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
  __sanitizer_syscall_post_futex(result, uaddr, futex_op, val, timeout, uaddr2,
                                 val3);
  return result;
}

// Simple mutex implementation using futex.
class Mutex {
public:
  Mutex() : value(0) {}

  void lock() {
    int c;
    while ((c = __sync_val_compare_and_swap(&value, 0, 1)) != 0) {
      if (c != 1)
        continue;
      int r = futex(&value, FUTEX_WAIT_PRIVATE, 1, nullptr, nullptr, 0);
      if (r == -1 && errno != EAGAIN) {
        fprintf(stderr, "futex wait error\n");
        abort();
      }
    }
  }

  void unlock() {
    value = 0;
    int r = futex(&value, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
    if (r == -1) {
      fprintf(stderr, "futex wake error\n");
      abort();
    }
  }

private:
  int value;
};

Mutex mutex;

void *Thread(void *x) {
  // Waiting for the futex.
  mutex.lock();
  // Finished waiting.
  return nullptr;
}

static void SigprofHandler(int signal, siginfo_t *info, void *context) {
  // Unlock the futex.
  mutex.unlock();
}

void InstallSignalHandler() {
  struct sigaction sa;
  sa.sa_sigaction = SigprofHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART | SA_SIGINFO;
  if (sigaction(SIGPROF, &sa, 0) != 0) {
    fprintf(stderr, "failed to install signal handler\n");
    abort();
  }
}

int main() {
  alarm(60); // Kill the test if it hangs.

  // Install the signal handler
  InstallSignalHandler();

  // Lock the futex at first so the other thread will wait for it.
  mutex.lock();

  // Create the thread to wait for the futex.
  pthread_t thread;
  pthread_create(&thread, NULL, Thread, NULL);

  // Just waiting a bit to make sure the thead is at the FUTEX_WAIT_PRIVATE
  // syscall.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Send the signal to the other thread, which will send the futex wake
  // syscall.
  int r = pthread_kill(thread, SIGPROF);
  assert(r == 0);

  // Futex should be notified and the thread should be able to continue.
  pthread_join(thread, NULL);

  // Exiting successfully.
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
