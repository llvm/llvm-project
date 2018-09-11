// Tests __hwasan_print_memory_usage through /proc/$PID/maps.
// RUN: %clang_hwasan %s -o %t && %env_hwasan_opts=export_memory_stats=1 %run %t 2>&1 | FileCheck %s
// REQUIRES: android

// This test requires a rooted device.
// UNSUPPORTED: android

#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
  char cmd[1024];
  snprintf(cmd, sizeof(cmd), "cat /proc/%d/maps", getpid());
  system(cmd);
  // CHECK: HWASAN pid: [[PID:[0-9]*]] rss: {{.*}} threads: 1 stacks: [[STACKS:[0-9]*]] thr_aux: {{.*}} stack_depot: {{.*}} uniq_stacks: [[UNIQ_STACKS:[0-9]*]] heap: [[HEAP:[0-9]*]]
}
