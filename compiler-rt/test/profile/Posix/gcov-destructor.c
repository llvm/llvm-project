// XFAIL: aix
/// Test that destructors and destructors whose priorities are greater than 100 are tracked.
// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: %clang --coverage %s -o %t
// RUN: rm -f gcov-destructor.gcda && %run %t
// RUN: llvm-cov gcov -t gcov-destructor.gcda | FileCheck %s
// UNSUPPORTED: darwin

#include <unistd.h>

void before_exec() {}                   // CHECK:      1: [[#@LINE]]:void before_exec
void after_exec() {}                    // CHECK-NEXT: 1: [[#@LINE]]:void after_exec

__attribute__((constructor))            // CHECK:      -: [[#@LINE]]:__attribute__
void constructor() {}                   // CHECK-NEXT: 1: [[#@LINE]]:

/// Runs before __llvm_gcov_writeout.
__attribute__((destructor))             // CHECK:      -: [[#@LINE]]:__attribute__
void destructor() {}                    // CHECK-NEXT: 1: [[#@LINE]]:

__attribute__((destructor(101)))        // CHECK:      -: [[#@LINE]]:__attribute__
void destructor_101() {}                // CHECK-NEXT: 1: [[#@LINE]]:

/// Runs after __llvm_gcov_writeout.
__attribute__((destructor(99)))         // CHECK:      -: [[#@LINE]]:__attribute__
void destructor_99() {}                 // CHECK-NEXT: #####: [[#@LINE]]:

int main() {
  before_exec();
  // Implicit writeout.
  execl("/not_exist", "not_exist", (char *)0);
  // Still tracked.
  after_exec();
  return 0;
}
