/// A basic block with fork/exec* is split. .gcda is flushed immediately before
/// fork/exec* so the lines before fork are counted once while succeeding
/// lines are counted twice.
// UNSUPPORTED: darwin

// RUN: mkdir -p %t.d && cd %t.d
// RUN: %clang --coverage %s -o %t
// RUN: test -f gcov-fork.gcno

// RUN: rm -f gcov-fork.gcda && %run %t
// RUN: llvm-cov gcov -t gcov-fork.gcda | FileCheck %s

#include <unistd.h>

void func1() {}                    // CHECK:      1: [[#@LINE]]:void func1()
void func2() {}                    // CHECK-NEXT: 2: [[#@LINE]]:
int main(void) {                   // CHECK-NEXT: 1: [[#@LINE]]:
  func1();                         // CHECK-NEXT: 1: [[#@LINE]]:
  fork();                          // CHECK-NEXT: 1: [[#@LINE]]:
  func2();                         // CHECK-NEXT: 2: [[#@LINE]]:
  return 0;                        // CHECK-NEXT: 2: [[#@LINE]]:
}
