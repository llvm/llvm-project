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
  // CHECK-UBSAN: %[[D:.*]] = load ptr, ptr %dest.addr
  // CHECK-UBSAN: %[[S:.*]] = load ptr, ptr %src.addr
  
  // Verify LHS (Dest) Check
  // CHECK-UBSAN: %[[D_NULL:.*]] = icmp ne ptr %[[D]], null
  // CHECK-UBSAN: %[[D_ALIGN:.*]] = and i64 %{{.*}}, 3
  // CHECK-UBSAN: %[[D_ALIGN_OK:.*]] = icmp eq i64 %[[D_ALIGN]], 0
  // CHECK-UBSAN: %[[D_OK:.*]] = and i1 %[[D_NULL]], %[[D_ALIGN_OK]]
  // CHECK-UBSAN: br i1 %[[D_OK]], label %[[D_CONT:.*]], label %[[D_HANDLER:.*]]

  // CHECK-UBSAN: [[D_HANDLER]]:
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort
  // CHECK-UBSAN: unreachable

  // CHECK-UBSAN: [[D_CONT]]:
  // Verify RHS (Src) Check
  // CHECK-UBSAN: %[[S_NULL:.*]] = icmp ne ptr %[[S]], null
  // CHECK-UBSAN: br i1 %[[S_NULL]], label %[[S_CONT:.*]], label %[[S_HANDLER:.*]]

  // CHECK-UBSAN: [[S_HANDLER]]:
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort

  // CHECK-UBSAN: [[S_CONT]]:
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[D]], ptr align 4 %[[S]], i64 4, i1 false)
  
  // CHECK-NO-UBSAN-NOT: @__ubsan_handle_type_mismatch
  *dest = *src;
}

// CHECK-LABEL: define {{.*}}void @test_nested_struct(
void test_nested_struct(struct Container *c, struct Small *s) {
  // CHECK-UBSAN: %[[C:.*]] = load ptr, ptr %c.addr
  // CHECK-UBSAN: icmp ne ptr %[[C]], null
  // CHECK-UBSAN: br i1 %{{.*}}, label %[[CONT:.*]], label %[[HANDLER:.*]]
  
  // CHECK-UBSAN: [[HANDLER]]:
  // CHECK-UBSAN: call void @__ubsan_handle_type_mismatch_v1_abort
  c->inner = *s;
}

#ifdef __cplusplus
}
#endif
