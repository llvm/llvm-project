// Ensure compatiblity of UBSan unreachable with ASan in the presence of
// noreturn functions.
// RUN: %clang_cc1 -fsanitize=unreachable,address        -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsanitize=unreachable,kernel-address -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s

void my_longjmp(void) __attribute__((noreturn));

// CHECK-LABEL: define void @calls_noreturn()
void calls_noreturn() {
  my_longjmp();
  // CHECK:      @__asan_handle_no_return{{.*}} !nosanitize
  // CHECK-NEXT: @my_longjmp(){{[^#]*}}
  // CHECK:      @__ubsan_handle_builtin_unreachable{{.*}} !nosanitize
  // CHECK-NEXT: unreachable
}

// CHECK: declare void @my_longjmp() [[FN_ATTR:#[0-9]+]]
// CHECK: declare void @__asan_handle_no_return()

// CHECK-LABEL: attributes
// CHECK-NOT: [[FN_ATTR]] = { {{.*noreturn.*}} }
