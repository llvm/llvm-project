// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - %s | FileCheck -check-prefixes CHECK,WITHOUT %s
// RUN: %clang -target x86_64-linux-gnu -S -emit-llvm -o - %s -fsanitize=thread | FileCheck -check-prefixes CHECK,TSAN %s

// Instrumented function.
// TSan inserts calls to __tsan_func_entry() and __tsan_func_exit() to prologue/epilogue.
// Non-atomic loads are instrumented with __tsan_readXXX(), atomic loads - with
// __tsan_atomicXXX_load().
//
// CHECK-LABEL: @instrumented1
// TSAN: call void @__tsan_func_entry
// WITHOUT-NOT: call void @__tsan_func_entry
// TSAN: call void @__tsan_read4
// WITHOUT-NOT: call void @__tsan_read4
// TSAN: call i32 @__tsan_atomic32_load
// WITHOUT-NOT: call i32 @__tsan_atomic32_load
// TSAN: call void @__tsan_func_exit
// WITHOUT-NOT: call void @__tsan_func_exit
// CHECK: ret i32
int instrumented1(int *a, _Atomic int *b) {
  return *a + *b;
}

// Function with no_sanitize("thread").
// TSan only inserts instrumentation necessary to prevent false positives: calls are inserted for
// function entry/exit and atomics, but not plain memory accesses.
//
// CHECK-LABEL: @no_false_positives1
// TSAN: call void @__tsan_func_entry
// WITHOUT-NOT: call void @__tsan_func_entry
// TSAN-NOT: call void @__tsan_read4
// WITHOUT-NOT: call void @__tsan_read4
// TSAN: call i32 @__tsan_atomic32_load
// WITHOUT-NOT: call i32 @__tsan_atomic32_load
// TSAN: call void @__tsan_func_exit
// WITHOUT-NOT: call void @__tsan_func_exit
// CHECK: ret i32
__attribute__((no_sanitize("thread"))) int no_false_positives1(int *a, _Atomic int *b) {
  return *a + *b;
}

// Function with disable_sanitizer_instrumentation: no instrumentation at all.
//
// CHECK-LABEL: @no_instrumentation1
// TSAN-NOT: call void @__tsan_func_entry
// WITHOUT-NOT: call void @__tsan_func_entry
// TSAN-NOT: call void @__tsan_read4
// WITHOUT-NOT: call void @__tsan_read4
// TSAN-NOT: call i32 @__tsan_atomic32_load
// WITHOUT-NOT: call i32 @__tsan_atomic32_load
// TSAN-NOT: call void @__tsan_func_exit
// WITHOUT-NOT: call void @__tsan_func_exit
// CHECK: ret i32
__attribute__((disable_sanitizer_instrumentation)) int no_instrumentation1(int *a, _Atomic int *b) {
  return *a + *b;
}
