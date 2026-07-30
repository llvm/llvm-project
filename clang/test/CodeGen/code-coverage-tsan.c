/// -fprofile-update=atomic requires the (potentially concurrent) counter updates to be atomic.
// RUN: %clang_cc1 %s -triple x86_64 -emit-llvm -fprofile-update=atomic \
// RUN:   -coverage-notes-file=/dev/null -coverage-data-file=/dev/null -o - | FileCheck %s

// CHECK-LABEL: void @foo()
/// Two counters are incremented by __tsan_atomic64_fetch_add.
// CHECK:         atomicrmw add ptr @__llvm_gcov_ctr{{.*}} monotonic, align 8
// CHECK-NEXT:    atomicrmw sub ptr

_Atomic(int) cnt;
void foo(void) { cnt--; }
