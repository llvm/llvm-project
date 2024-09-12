// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// Test __atomic_* built-ins that have a memory order parameter with a runtime
// value.  This requires generating a switch statement, so the amount of
// generated code is surprisingly large.
//
// This is just a quick smoke test.  Only atomic_load_n is tested.

int runtime_load(int *ptr, int order) {
  return __atomic_load_n(ptr, order);
}

// CHECK:   %[[T7:[0-9]+]] = load ptr, ptr %[[T3:[0-9]+]], align 8
// CHECK:   %[[T8:[0-9]+]] = load i32, ptr %[[T4:[0-9]+]], align 4
// CHECK:   switch i32 %[[T8]], label %[[L9:[0-9]+]] [
// CHECK:     i32 1, label %[[L11:[0-9]+]]
// CHECK:     i32 2, label %[[L11]]
// CHECK:     i32 5, label %[[L13:[0-9]+]]
// CHECK:   ]
// CHECK: [[L9]]:
// CHECK:   %[[T10:[0-9]+]] = load atomic i32, ptr %[[T7]] monotonic, align 4
// CHECK:   store i32 %[[T10]], ptr %[[T6:[0-9]+]], align 4
// CHECK:   br label %[[L15:[0-9]+]]
// CHECK: [[L11]]:
// CHECK:   %[[T12:[0-9]+]] = load atomic i32, ptr %[[T7]] acquire, align 4
// CHECK:   store i32 %[[T12]], ptr %[[T6]], align 4
// CHECK:   br label %[[L15]]
// CHECK: [[L13]]:
// CHECK:   %[[T14:[0-9]+]] = load atomic i32, ptr %[[T7]] seq_cst, align 4
// CHECK:   store i32 %[[T14]], ptr %[[T6]], align 4
// CHECK:   br label %[[L15]]
// CHECK: [[L15]]:
// CHECK:   %[[T16:[0-9]+]] = load i32, ptr %[[T6]], align 4
// CHECK:   store i32 %[[T16]], ptr %[[T5:[0-9]+]], align 4
// CHECK:   %[[T17:[0-9]+]] = load i32, ptr %[[T5]], align 4
// CHECK:   ret i32 %[[T17]]
