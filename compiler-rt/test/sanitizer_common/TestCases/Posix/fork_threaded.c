// RUN: %clang -O0 %s -o %t && %env_tool_opts=die_after_fork=0 %run %t

// The test uses pthread barriers which are not available on Darwin.
// UNSUPPORTED: darwin

// FIXME: It probably hangs on this platform.
// UNSUPPORTED: ppc

// FIXME: TSAN does not lock allocator.
// UNSUPPORTED: tsan

// FIXME: False stack overflow report
// UNSUPPORTED: android && asan

// FIXME: Requires `FutexWait` implementation. See __asan::InstallAtForkHandler.
// UNSUPPORTED: target={{.*solaris.*}}
// UNSUPPORTED: target={{.*netbsd.*}}
// UNSUPPORTED: target={{.*apple.*}}

// Forking in multithread environment is unsupported. However we already have
// some workarounds, and will add more, so this is the test.
// The test try to check two things:
//  1. Internal mutexes used by `inparent` thread do not deadlock `inchild`
//     thread.
//  2. Stack poisoned by `inparent` is not poisoned in `inchild` thread.

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sanitizer_common/sanitizer_specific.h"

static const size_t kBufferSize = 8192;

pthread_barrier_t bar;

// Without appropriate workarounds this code can cause the forked process to
// start with locked internal mutexes.
void ShouldNotDeadlock() {
  // Don't bother with leaks, we try to trigger allocator or lsan deadlock.
  __lsan_disable();
  void *volatile p = malloc(10);
  __lsan_do_recoverable_leak_check();
  // Allocator still in broken state, `free` may report errors.
  // free(p);
  __lsan_enable();
}

// Prevent stack buffer cleanup by instrumentation.
#define NOSAN __attribute__((no_sanitize("address", "hwaddress", "memory")))

NOSAN static void *inparent(void *arg) {
  char t[kBufferSize];
  make_mem_bad(t, sizeof(t));

  pthread_barrier_wait(&bar);

  for (;;)
    ShouldNotDeadlock();

  return 0;
}

NOSAN static void *inchild(void *arg) {
  char t[kBufferSize];
  check_mem_is_good(t, sizeof(t));
  ShouldNotDeadlock();
  return 0;
}

int main(void) {
#if __has_feature(hwaddress_sanitizer)
  __hwasan_enable_allocator_tagging();
#endif

  pid_t pid;

  pthread_barrier_init(&bar, NULL, 2);
  pthread_t thread_id;
  while (pthread_create(&thread_id, 0, &inparent, 0) != 0) {
  }
  pthread_barrier_wait(&bar);

  pid = fork();
  switch (pid) {
  case -1:
    perror("fork");
    return -1;
  case 0:
    ShouldNotDeadlock();
    while (pthread_create(&thread_id, 0, &inchild, 0) != 0) {
    }
    pthread_join(thread_id, NULL);
    _exit(0);
    break;
  default: {
    int status;
    pid_t child = waitpid(pid, &status, /*options=*/0);
    assert(pid == child);
    assert(WIFEXITED(status));
    assert(WEXITSTATUS(status) == 0);
    break;
  }
  }

  return 0;
}
