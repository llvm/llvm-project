// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard %s -o %t
// RUN: rm -rf %T/coverage-fork
// RUN: mkdir -p %T/coverage-fork && cd %T/coverage-fork
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: android
//
// Ideally a forked-subprocess should only report it's own coverage,
// not parent's one. But trace-pc-guard currently does nothing special for fork,
// and thus this test is relaxed.

#include <stdio.h>
#include <string.h>
#include <unistd.h>

__attribute__((noinline))
void foo() { printf("foo\n"); }

__attribute__((noinline))
void bar() { printf("bar\n"); }

__attribute__((noinline))
void baz() { printf("baz\n"); }

int main(int argc, char **argv) {
  pid_t child_pid = fork();
  if (child_pid == 0) {
    fprintf(stderr, "Child PID: %d\n", getpid());
    baz();
  } else {
    fprintf(stderr, "Parent PID: %d\n", getpid());
    foo();
    bar();
  }
  return 0;
}

// CHECK-DAG: Child PID: [[ChildPID:[0-9]+]]
// CHECK-DAG: [[ChildPID]].sancov: {{.*}} PCs written
// CHECK-DAG: Parent PID: [[ParentPID:[0-9]+]]
// CHECK-DAG: [[ParentPID]].sancov: 3 PCs written
