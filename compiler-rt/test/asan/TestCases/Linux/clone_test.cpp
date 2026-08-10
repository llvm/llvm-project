// Regression test for:
// http://code.google.com/p/address-sanitizer/issues/detail?id=37

// RUN: %clangxx_asan -O0 %s -o %t && %run %t | FileCheck %s
// RUN: %clangxx_asan -O1 %s -o %t && %run %t | FileCheck %s
// RUN: %clangxx_asan -O2 %s -o %t && %run %t | FileCheck %s
// RUN: %clangxx_asan -O3 %s -o %t && %run %t | FileCheck %s

#include <stdio.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int Child(void *arg) {
  char x[32] = {0};  // Stack gets poisoned.
  printf("Child:  %p\n", x);
  _exit(1);  // NoReturn, stack will remain unpoisoned unless we do something.
}

int main(int argc, char **argv) {
  const int kStackSize = 1 << 20;
  char __attribute__((aligned(16))) child_stack[kStackSize + 1];
  char *sp = child_stack + kStackSize;  // Stack grows down.
  printf("Parent: %p\n", sp);
  pid_t clone_pid = clone(Child, sp, CLONE_FILES | CLONE_VM, NULL);
  int status;
  pid_t wait_result = waitpid(clone_pid, &status, __WCLONE);
  if (wait_result < 0) {
    perror("waitpid");
    return 0;
  }
  if (wait_result == clone_pid && WIFEXITED(status)) {
    // Make sure the child stack was indeed unpoisoned.
    for (int i = 0; i < kStackSize; i++)
      child_stack[i] = i;
    int ret = child_stack[argc - 1];
    printf("PASSED\n");
    // CHECK: PASSED
    return ret;
  }
  return 0;
}
