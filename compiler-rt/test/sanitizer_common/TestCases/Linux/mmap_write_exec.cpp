// RUN: %clangxx %s -o %t
// RUN: %env_tool_opts=detect_write_exec=1 %run %t 2>&1 | FileCheck %s
// RUN: %env_tool_opts=detect_write_exec=0 %run %t 2>&1 | FileCheck %s \
// RUN:   --check-prefix=CHECK-DISABLED
// ubsan and lsan do not install mmap interceptors UNSUPPORTED: ubsan, lsan

// TODO: Fix option on Android, it hangs there for unknown reasons.
// XFAIL: android

#include <stdio.h>
#include <sys/mman.h>

int main(int argc, char **argv) {
  char *p = (char *)mmap(0, 1024, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  // CHECK: WARNING: {{.*}}Sanitizer: writable-executable page usage
  // CHECK: #{{[0-9]+.*}}main{{.*}}mmap_write_exec.cpp:[[@LINE-3]]
  // CHECK: SUMMARY: {{.*}}Sanitizer: w-and-x-usage

  char *q = (char *)mmap(p, 64, PROT_READ | PROT_WRITE,
                         MAP_ANONYMOUS | MAP_PRIVATE | MAP_FIXED, -1, 0);
  (void)mprotect(q, 64, PROT_WRITE | PROT_EXEC);
  // CHECK: WARNING: {{.*}}Sanitizer: writable-executable page usage
  // CHECK: #{{[0-9]+.*}}main{{.*}}mmap_write_exec.cpp:[[@LINE-2]]
  // CHECK: SUMMARY: {{.*}}Sanitizer: w-and-x-usage

  char *a = (char *)mmap(0, 1024, PROT_READ | PROT_WRITE,
                         MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  char *b = (char *)mmap(a, 64, PROT_READ | PROT_WRITE,
                         MAP_ANONYMOUS | MAP_PRIVATE | MAP_FIXED, -1, 0);
  (void)mprotect(q, 64, PROT_READ | PROT_EXEC);
  // CHECK-NOT: Sanitizer

  printf("done\n");
  // CHECK-DISABLED-NOT: Sanitizer
  // CHECK-DISABLED: done
}
