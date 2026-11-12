// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=unsigned-integer-overflow -fsanitize-recover=unsigned-integer-overflow %s -emit-llvm -o - | FileCheck %s --check-prefix=RECOVER
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=unsigned-integer-overflow %s -emit-llvm -o - | FileCheck %s --check-prefix=ABORT
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=null,object-size,alignment -fsanitize-recover=object-size %s -emit-llvm -o - | FileCheck %s --check-prefix=PARTIAL

// RECOVER: @test
// ABORT: @test
void test(void) {
  extern volatile unsigned x, y, z;

  // RECOVER: uadd.with.overflow.i32{{.*}}, !nosanitize
  // RECOVER: ubsan_handle_add_overflow({{.*}}, !nosanitize
  // RECOVER-NOT: unreachable
  // ABORT: uadd.with.overflow.i32{{.*}}, !nosanitize
  // ABORT: ubsan_handle_add_overflow_abort({{.*}}, !nosanitize
  // ABORT: unreachable{{.*}}, !nosanitize
  x = y + z;
}

void foo(void) {
  union { int i; } u;
  u.i=1;
  // PARTIAL:      %[[SIZE:.*]] = call i64 @llvm.objectsize.i64.p0(ptr {{.*}}, i1 false, i1 false)
  // PARTIAL-NEXT: %[[CHECK0:.*]] = icmp uge i64 %[[SIZE]], 4

  // PARTIAL:      br i1 %[[CHECK0]], {{.*}} !nosanitize

  // PARTIAL:      call void @__ubsan_handle_type_mismatch_v1(
}
