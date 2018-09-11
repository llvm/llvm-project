// Tests __hwasan_print_memory_usage through /proc/$PID/maps.
// RUN: %clang_hwasan %s -o %t && %env_hwasan_opts=export_memory_stats=1 %run %t 2>&1 | FileCheck %s
// REQUIRES: android

#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

// The function needs to be unsanitized in order for &cmd to be untagged. This
// address is passed to system() and then to execve() syscall. The tests need to
// run on unpatched linux kernel, which at this time does not accept tagged
// pointers in system call arguments (but there is hope: see
// https://lore.kernel.org/patchwork/cover/979328).
__attribute__((no_sanitize("hwaddress")))
int main() {
  char cmd[1024];
  snprintf(cmd, sizeof(cmd), "cat /proc/%d/maps", getpid());
  system(cmd);
  // CHECK: HWASAN pid: [[PID:[0-9]*]] rss: {{.*}} threads: 1 stacks: [[STACKS:[0-9]*]] thr_aux: {{.*}} stack_depot: {{.*}} uniq_stacks: [[UNIQ_STACKS:[0-9]*]] heap: [[HEAP:[0-9]*]]
}
