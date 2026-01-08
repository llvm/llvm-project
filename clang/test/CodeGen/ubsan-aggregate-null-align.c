// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - \
// RUN:    -fsanitize=null,alignment | FileCheck %s --check-prefix=CHECK-SANITIZE
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - \
// RUN:    | FileCheck %s --check-prefix=CHECK-NO-SANITIZE

struct Small { int x; };
struct Container { struct Small inner; };

// CHECK-SANITIZE-LABEL: define {{.*}}void @test_direct_assign_ptr(
// CHECK-SANITIZE: %[[D:.*]] = load ptr, ptr %dest.addr
// CHECK-SANITIZE: %[[S:.*]] = load ptr, ptr %src.addr
// CHECK-SANITIZE: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[D]], ptr align 4 %[[S]], i64 4, i1 false)
//
// CHECK-NO-SANITIZE-LABEL: define {{.*}}void @test_direct_assign_ptr(
// CHECK-NO-SANITIZE-NOT: @__ubsan_handle_type_mismatch
void test_direct_assign_ptr(struct Small *dest, struct Small *src) {
  *dest = *src;
}

// CHECK-SANITIZE-LABEL: define {{.*}}void @test_null_dest(
// CHECK-SANITIZE: %[[D:.*]] = load ptr, ptr %dest
// CHECK-SANITIZE: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[D]], ptr {{.*}}, i64 4, i1 false)
//
// CHECK-NO-SANITIZE-LABEL: define {{.*}}void @test_null_dest(
// CHECK-NO-SANITIZE-NOT: @__ubsan_handle_type_mismatch
void test_null_dest(struct Small *src) {
  struct Small *dest = 0;
  *dest = *src;
}

// CHECK-SANITIZE-LABEL: define {{.*}}void @test_nested_struct(
// CHECK-SANITIZE: %[[VAL1:.*]] = icmp ne ptr %[[C:.*]], null
// CHECK-SANITIZE: br i1 %{{.*}}, label %cont, label %handler.type_mismatch
//
// CHECK-NO-SANITIZE-LABEL: define {{.*}}void @test_nested_struct(
// CHECK-NO-SANITIZE-NOT: @__ubsan_handle_type_mismatch
void test_nested_struct(struct Container *c, struct Small *s) {
  c->inner = *s;
}

// CHECK-SANITIZE-LABEL: define {{.*}}void @test_comma_operator(
// CHECK-SANITIZE: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 4, i1 false)
//
// CHECK-NO-SANITIZE-LABEL: define {{.*}}void @test_comma_operator(
// CHECK-NO-SANITIZE-NOT: @__ubsan_handle_type_mismatch
void test_comma_operator(struct Small *dest, struct Small *src) {
  *dest = (0, *src);
}
