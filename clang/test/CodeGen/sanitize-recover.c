// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=unsigned-integer-overflow -fsanitize-recover=unsigned-integer-overflow %s -emit-llvm -o - | FileCheck %s --check-prefix=RECOVER
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=unsigned-integer-overflow %s -emit-llvm -o - | FileCheck %s --check-prefix=ABORT
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=null,object-size,alignment -fsanitize-recover=object-size %s -emit-llvm -o - | FileCheck %s --check-prefix=PARTIAL

// RECOVER: @test
// ABORT: @test
void test() {
  extern volatile unsigned x, y, z;

  // RECOVER: uadd.with.overflow.i32
  // RECOVER: ubsan_handle_add_overflow(
  // RECOVER-NOT: unreachable
  // ABORT: uadd.with.overflow.i32
  // ABORT: ubsan_handle_add_overflow_abort(
  // ABORT: unreachable
  x = y + z;
}

void foo() {
  union { int i; } u;
  u.i=1;
  // PARTIAL:      %[[CHECK0:.*]] = icmp ne {{.*}}* %[[PTR:.*]], null

  // PARTIAL:      %[[SIZE:.*]] = call i64 @llvm.objectsize.i64.p0i8(i8* {{.*}}, i1 false)
  // PARTIAL-NEXT: %[[CHECK1:.*]] = icmp uge i64 %[[SIZE]], 4

  // PARTIAL:      %[[MISALIGN:.*]] = and i64 {{.*}}, 3
  // PARTIAL-NEXT: %[[CHECK2:.*]] = icmp eq i64 %[[MISALIGN]], 0

  // PARTIAL:      %[[CHECK02:.*]] = and i1 %[[CHECK0]], %[[CHECK2]]
  // PARTIAL-NEXT: %[[CHECK012:.*]] = and i1 %[[CHECK02]], %[[CHECK1]]

  // PARTIAL:      br i1 %[[CHECK012]], {{.*}} !prof ![[WEIGHT_MD:.*]], !nosanitize

  // PARTIAL:      br i1 %[[CHECK02]], {{.*}}
  // PARTIAL:      call void @__ubsan_handle_type_mismatch_v1_abort(
  // PARTIAL-NEXT: unreachable
  // PARTIAL:      call void @__ubsan_handle_type_mismatch_v1(
}
