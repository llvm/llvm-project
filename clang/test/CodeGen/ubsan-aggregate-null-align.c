// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=alignment,null \
// RUN:   -emit-llvm -std=c23 %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-UBSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -std=c23 %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-NO-UBSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=alignment,null \
// RUN:   -emit-llvm -xc++ %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-UBSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -xc++ %s -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-NO-UBSAN

typedef struct Small { int x; } Small;
typedef struct Container { struct Small inner; } Container;

#ifdef __cplusplus
extern "C" {
#endif

// CHECK-LABEL: define {{.*}}void @test_direct_assign_ptr(
void test_direct_assign_ptr(struct Small *dest, struct Small *src) {
  // CHECK-UBSAN: icmp ne ptr {{.*}}, null
  // CHECK-UBSAN: and i64
  // CHECK-UBSAN: icmp eq i64
  // CHECK-UBSAN: and i1
  // CHECK-UBSAN: br i1 {{.*}}, label %cont, label %handler.type_mismatch
  // CHECK-UBSAN: handler.type_mismatch:
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort
  // CHECK-UBSAN: unreachable
  // CHECK-UBSAN: cont:
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 {{.*}}, ptr align 4 {{.*}}, i64 4, i1 false)
  // CHECK-NO-UBSAN-NOT: @__ubsan_handle_type_mismatch

  *dest = *src;
}

// CHECK-LABEL: define {{.*}}void @test_nested_struct(
void test_nested_struct(struct Container *c, struct Small *s) {
  // CHECK-UBSAN: icmp ne ptr {{.*}}, null
  // CHECK-UBSAN: br i1 {{.*}}, label %cont, label %handler.type_mismatch
  // CHECK-UBSAN: handler.type_mismatch:
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort

  c->inner = *s;
}

#ifdef __cplusplus
}
#endif
