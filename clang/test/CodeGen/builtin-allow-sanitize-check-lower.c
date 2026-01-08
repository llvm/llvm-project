// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=address -emit-llvm -O0 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=address -emit-llvm -O1 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=address -emit-llvm -O2 -o - %s | FileCheck %s

// CHECK-NOT: call{{.*}} @llvm.allow.sanitize.address

__attribute__((always_inline))
_Bool check() {
  return __builtin_allow_sanitize_check("address");
}

// CHECK-LABEL: @test_sanitize
// CHECK: ret i1 true
_Bool test_sanitize() {
  return check();
}

// CHECK-LABEL: @test_no_sanitize
// CHECK: ret i1 false
__attribute__((no_sanitize("address")))
_Bool test_no_sanitize() {
  return check();
}
