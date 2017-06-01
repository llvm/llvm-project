// RUN: %clangxx_asan -fsanitize-coverage=func %s -o %t
// RUN: rm -rf %T/coverage-fork
// RUN: mkdir -p %T/coverage-fork && cd %T/coverage-fork
// RUN: %env_asan_opts=coverage=1:coverage_direct=0:verbosity=1 %run %t 2>&1 | FileCheck %s
//
// UNSUPPORTED: android

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
// CHECK-DAG: [[ChildPID]].sancov: 1 PCs written
// CHECK-DAG: Parent PID: [[ParentPID:[0-9]+]]
// CHECK-DAG: [[ParentPID]].sancov: 3 PCs written
