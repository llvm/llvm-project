// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=unsigned-integer-overflow %s -emit-llvm -o - | FileCheck %s
// Verify checked operations are emitted for integers and longs.
// unsigned short/char's tested in unsigned-promotion.c

unsigned long li, lj, lk;
unsigned int ii, ij, ik;

// The wraps attribute disables sanitizer instrumentation for arithmetic
// expressions containing these types.
unsigned long __attribute__((wraps)) li_w, lj_w, lk_w;
unsigned int __attribute__((wraps)) ii_w, ij_w, ik_w;

extern void opaquelong(unsigned long);
extern void opaqueint(unsigned int);

// CHECK-LABEL: define{{.*}} void @testlongadd()
void testlongadd(void) {

  // CHECK:      [[T1:%.*]] = load i64, ptr @lj
  // CHECK-NEXT: [[T2:%.*]] = load i64, ptr @lk
  // CHECK-NEXT: [[T3:%.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 [[T1]], i64 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i64, i1 } [[T3]], 1
  // CHECK: call void @__ubsan_handle_add_overflow
  li = lj + lk;

  // CHECK: [[T6:%.*]] = load i64, ptr @lj_w
  // CHECK-NEXT: [[T7:%.*]] = load i64, ptr @lk_w
  // CHECK-NEXT: add i64 [[T6]], [[T7]]
  li_w = lj_w + lk_w;
}

// CHECK-LABEL: define{{.*}} void @testlongsub()
void testlongsub(void) {

  // CHECK:      [[T1:%.*]] = load i64, ptr @lj
  // CHECK-NEXT: [[T2:%.*]] = load i64, ptr @lk
  // CHECK-NEXT: [[T3:%.*]] = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 [[T1]], i64 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i64, i1 } [[T3]], 1
  // CHECK: call void @__ubsan_handle_sub_overflow
  li = lj - lk;

  // CHECK: [[T6:%.*]] = load i64, ptr @lj_w
  // CHECK-NEXT: [[T7:%.*]] = load i64, ptr @lk_w
  // CHECK-NEXT: sub i64 [[T6]], [[T7]]
  li_w = lj_w - lk_w;
}

// CHECK-LABEL: define{{.*}} void @testlongmul()
void testlongmul(void) {

  // CHECK:      [[T1:%.*]] = load i64, ptr @lj
  // CHECK-NEXT: [[T2:%.*]] = load i64, ptr @lk
  // CHECK-NEXT: [[T3:%.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 [[T1]], i64 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i64, i1 } [[T3]], 1
  // CHECK: call void @__ubsan_handle_mul_overflow
  li = lj * lk;

  // CHECK: [[T6:%.*]] = load i64, ptr @lj_w
  // CHECK-NEXT: [[T7:%.*]] = load i64, ptr @lk_w
  // CHECK-NEXT: mul i64 [[T6]], [[T7]]
  li_w = lj_w * lk_w;
}

// CHECK-LABEL: define{{.*}} void @testlongpostinc()
void testlongpostinc(void) {
  // CHECK:      [[T1:%.*]] = load i64, ptr @li
  // CHECK-NEXT: [[T2:%.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 [[T1]], i64 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i64, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T2]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
  opaquelong(li++);

  // CHECK: [[T5:%.*]] = load i64, ptr @li_w
  // CHECK-NEXT: add i64 [[T5]], 1
  opaquelong(li_w++);
}

// CHECK-LABEL: define{{.*}} void @testlongpreinc()
void testlongpreinc(void) {
  // CHECK:      [[T1:%.*]] = load i64, ptr @li
  // CHECK-NEXT: [[T2:%.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 [[T1]], i64 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i64, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T2]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
  opaquelong(++li);

  // CHECK: [[T5:%.*]] = load i64, ptr @li_w
  // CHECK-NEXT: add i64 [[T5]], 1
  opaquelong(++li_w);
}

// CHECK-LABEL: define{{.*}} void @testintadd()
void testintadd(void) {

  // CHECK:      [[T1:%.*]] = load i32, ptr @ij
  // CHECK-NEXT: [[T2:%.*]] = load i32, ptr @ik
  // CHECK-NEXT: [[T3:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[T1]], i32 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i32, i1 } [[T3]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
  ii = ij + ik;

  // CHECK: [[T6:%.*]] = load i32, ptr @ij_w
  // CHECK-NEXT: [[T7:%.*]] = load i32, ptr @ik_w
  // CHECK-NEXT: add i32 [[T6]], [[T7]]
  ii_w = ij_w + ik_w;
}

// CHECK-LABEL: define{{.*}} void @testintsub()
void testintsub(void) {

  // CHECK:      [[T1:%.*]] = load i32, ptr @ij
  // CHECK-NEXT: [[T2:%.*]] = load i32, ptr @ik
  // CHECK-NEXT: [[T3:%.*]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 [[T1]], i32 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i32, i1 } [[T3]], 1
  // CHECK:      call void @__ubsan_handle_sub_overflow
  ii = ij - ik;

  // CHECK: [[T6:%.*]] = load i32, ptr @ij_w
  // CHECK-NEXT: [[T7:%.*]] = load i32, ptr @ik_w
  // CHECK-NEXT: sub i32 [[T6]], [[T7]]
  ii_w = ij_w - ik_w;
}

// CHECK-LABEL: define{{.*}} void @testintmul()
void testintmul(void) {

  // CHECK:      [[T1:%.*]] = load i32, ptr @ij
  // CHECK-NEXT: [[T2:%.*]] = load i32, ptr @ik
  // CHECK-NEXT: [[T3:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 [[T1]], i32 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i32, i1 } [[T3]], 1
  // CHECK:      call void @__ubsan_handle_mul_overflow
  ii = ij * ik;

  // CHECK: [[T6:%.*]] = load i32, ptr @ij_w
  // CHECK-NEXT: [[T7:%.*]] = load i32, ptr @ik_w
  // CHECK-NEXT: mul i32 [[T6]], [[T7]]
  ii_w = ij_w * ik_w;
}

// CHECK-LABEL: define{{.*}} void @testintpostinc()
void testintpostinc(void) {
  // CHECK:      [[T1:%.*]] = load i32, ptr @ii
  // CHECK-NEXT: [[T2:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i32, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T2]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
  opaqueint(ii++);

  // CHECK: [[T5:%.*]] = load i32, ptr @ii_w
  // CHECK-NEXT: add i32 [[T5]], 1
  opaqueint(ii_w++);
}

// CHECK-LABEL: define{{.*}} void @testintpreinc()
void testintpreinc(void) {
  // CHECK:      [[T1:%.*]] = load i32, ptr @ii
  // CHECK-NEXT: [[T2:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i32, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T2]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
  opaqueint(++ii);

  // CHECK: [[T5:%.*]] = load i32, ptr @ii_w
  // CHECK-NEXT: add i32 [[T5]], 1
  opaqueint(++ii_w);
}
