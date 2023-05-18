// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-linux %s \
// RUN:     | FileCheck %s --check-prefixes=CHECK-RUNTIME \
// RUN:         --implicit-check-not="__asan_stack_malloc_always_"
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-linux %s \
// RUN:         -fsanitize-address-use-after-return=runtime \
// RUN:     | FileCheck %s --check-prefixes=CHECK-RUNTIME \
// RUN:         --implicit-check-not="__asan_stack_malloc_always_"
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-linux %s \
// RUN:         -fsanitize-address-use-after-return=always \
// RUN:     | FileCheck %s --check-prefixes=CHECK-ALWAYS \
// RUN:         --implicit-check-not=__asan_option_detect_stack_use_after_return \
// RUN:         --implicit-check-not="__asan_stack_malloc_{{[0-9]}}"
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-linux %s \
// RUN:         -fsanitize-address-use-after-return=never \
// RUN:     | FileCheck %s \
// RUN:         --implicit-check-not=__asan_option_detect_stack_use_after_return \
// RUN:         --implicit-check-not="__asan_stack_malloc_"

// CHECK-RUNTIME: load{{.*}}@__asan_option_detect_stack_use_after_return
// CHECK-RUNTIME: call{{.*}}__asan_stack_malloc_0
// CHECK-ALWAYS: call{{.*}}__asan_stack_malloc_always_0

int *function1() {
  int x = 0;

#pragma clang diagnostic ignored "-Wreturn-stack-address"
  return &x;
}

int main() {
  auto px = function1();
  return 0;
}
