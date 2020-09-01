/// -fsanitize=thread requires the (potentially concurrent) counter updates to be atomic.
// RUN: %clang_cc1 %s -triple x86_64 -emit-llvm -fsanitize=thread -femit-coverage-notes -femit-coverage-data \
// RUN:   -coverage-notes-file /dev/null -coverage-data-file /dev/null -o - | FileCheck %s

// CHECK-LABEL: void @foo()
/// Two counters are incremented by __tsan_atomic64_fetch_add.
// CHECK:         call i64 @__tsan_atomic64_fetch_add
// CHECK-NEXT:    call i64 @__tsan_atomic64_fetch_add
// CHECK-NEXT:    call i32 @__tsan_atomic32_fetch_sub

_Atomic(int) cnt;
void foo() { cnt--; }
