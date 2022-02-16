/// -fprofile-update=atomic (implied by -fsanitize=thread) requires the
/// (potentially concurrent) counter updates to be atomic.
// RUN: %clang_cc1 %s -triple x86_64 -emit-llvm -fprofile-update=atomic -ftest-coverage -fprofile-arcs \
// RUN:   -coverage-notes-file /dev/null -coverage-data-file /dev/null -o - | FileCheck %s

// CHECK-LABEL: void @foo()
/// Two counters are incremented by __tsan_atomic64_fetch_add.
// CHECK:         atomicrmw add i64* {{.*}} @__llvm_gcov_ctr{{.*}} monotonic, align 8
// CHECK-NEXT:    atomicrmw sub i32*

_Atomic(int) cnt;
void foo(void) { cnt--; }
