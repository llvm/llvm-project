// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// REQUIRES: linux
#include "test.h"

#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

long count_memory_mappings() {
  pid_t my_pid = getpid();
  char proc_file_name[128];
  snprintf(proc_file_name, sizeof(proc_file_name), "/proc/%d/maps", my_pid);

  FILE *proc_file = fopen(proc_file_name, "r");
  long line_count = 0;
  int c;
  do {
    c = fgetc(proc_file);
    if (c == '\n') {
      line_count++;
    }
  } while (c != EOF);
  fclose(proc_file);

  return line_count;
}

void fiber_iteration() {
  void *orig_fiber = __tsan_get_current_fiber();
  void *fiber = __tsan_create_fiber(0);

  pthread_mutex_t mutex;
  pthread_mutex_init(&mutex, NULL);

  // Running some code on the fiber that triggers handling of pending signals.
  __tsan_switch_to_fiber(fiber, 0);
  pthread_mutex_lock(&mutex);
  pthread_mutex_unlock(&mutex);
  __tsan_switch_to_fiber(orig_fiber, 0);

  // We expect the fiber to clean up all resources (here the sigcontext) when destroyed.
  __tsan_destroy_fiber(fiber);
}

// Magic-Number for some warmup iterations,
// as tsan maps some memory for the first runs.
const size_t num_warmup = 100;

int main() {
  for (size_t i = 0; i < num_warmup; i++) {
    fiber_iteration();
  }

  long memory_mappings_before = count_memory_mappings();
  fiber_iteration();
  fiber_iteration();
  long memory_mappings_after = count_memory_mappings();

  // Is there a better way to detect a resource  leak in the
  // ThreadState object? (i.e. a mmap not being freed)
  if (memory_mappings_before == memory_mappings_after) {
    fprintf(stderr, "PASS\n");
  } else {
    fprintf(stderr, "FAILED\n");
  }

  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
