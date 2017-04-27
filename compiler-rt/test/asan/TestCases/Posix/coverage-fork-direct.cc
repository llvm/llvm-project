// RUN: %clangxx_asan -fsanitize-coverage=func %s -o %t
// RUN: rm -rf %T/coverage-fork-direct
// RUN: mkdir -p %T/coverage-fork-direct && cd %T/coverage-fork-direct
// RUN: %env_asan_opts=coverage=1:coverage_direct=1:verbosity=1 %run %t > %t.log 2>&1
// RUN: %sancov rawunpack *.sancov.raw
// RUN: %sancov print *.sancov >> %t.log 2>&1
// RUN: FileCheck %s < %t.log
//
// XFAIL: android

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
// CHECK-DAG: Parent PID: [[ParentPID:[0-9]+]]
// CHECK-DAG: read 3 {{64|32}}-bit PCs from {{.*}}.[[ParentPID]].sancov

// FIXME: this is missing
// XCHECK-DAG: read 1 {{64|32}}-bit PCs from {{.*}}.[[ChildPID]].sancov
