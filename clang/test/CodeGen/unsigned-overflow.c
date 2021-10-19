// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=unsigned-integer-overflow %s -emit-llvm -o - | FileCheck %s
// Verify checked operations are emitted for integers and longs.
// unsigned short/char's tested in unsigned-promotion.c

unsigned long li, lj, lk;
unsigned int ii, ij, ik;

extern void opaquelong(unsigned long);
extern void opaqueint(unsigned int);

// CHECK-LABEL: define{{.*}} void @testlongadd()
void testlongadd() {

  // CHECK:      [[T1:%.*]] = load i64, i64* @lj
  // CHECK-NEXT: [[T2:%.*]] = load i64, i64* @lk
  // CHECK-NEXT: [[T3:%.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 [[T1]], i64 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i64, i1 } [[T3]], 1
  // CHECK: call void @__ubsan_handle_add_overflow
  li = lj + lk;
}

// CHECK-LABEL: define{{.*}} void @testlongsub()
void testlongsub() {

  // CHECK:      [[T1:%.*]] = load i64, i64* @lj
  // CHECK-NEXT: [[T2:%.*]] = load i64, i64* @lk
  // CHECK-NEXT: [[T3:%.*]] = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 [[T1]], i64 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i64, i1 } [[T3]], 1
  // CHECK: call void @__ubsan_handle_sub_overflow
  li = lj - lk;
}

// CHECK-LABEL: define{{.*}} void @testlongmul()
void testlongmul() {

  // CHECK:      [[T1:%.*]] = load i64, i64* @lj
  // CHECK-NEXT: [[T2:%.*]] = load i64, i64* @lk
  // CHECK-NEXT: [[T3:%.*]] = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 [[T1]], i64 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i64, i1 } [[T3]], 1
  // CHECK: call void @__ubsan_handle_mul_overflow
  li = lj * lk;
}

// CHECK-LABEL: define{{.*}} void @testlongpostinc()
void testlongpostinc() {
  opaquelong(li++);

  // CHECK:      [[T1:%.*]] = load i64, i64* @li
  // CHECK-NEXT: [[T2:%.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 [[T1]], i64 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i64, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T2]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
}

// CHECK-LABEL: define{{.*}} void @testlongpreinc()
void testlongpreinc() {
  opaquelong(++li);

  // CHECK:      [[T1:%.*]] = load i64, i64* @li
  // CHECK-NEXT: [[T2:%.*]] = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 [[T1]], i64 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i64, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i64, i1 } [[T2]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
}

// CHECK-LABEL: define{{.*}} void @testintadd()
void testintadd() {

  // CHECK:      [[T1:%.*]] = load i32, i32* @ij
  // CHECK-NEXT: [[T2:%.*]] = load i32, i32* @ik
  // CHECK-NEXT: [[T3:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[T1]], i32 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i32, i1 } [[T3]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
  ii = ij + ik;
}

// CHECK-LABEL: define{{.*}} void @testintsub()
void testintsub() {

  // CHECK:      [[T1:%.*]] = load i32, i32* @ij
  // CHECK-NEXT: [[T2:%.*]] = load i32, i32* @ik
  // CHECK-NEXT: [[T3:%.*]] = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 [[T1]], i32 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i32, i1 } [[T3]], 1
  // CHECK:      call void @__ubsan_handle_sub_overflow
  ii = ij - ik;
}

// CHECK-LABEL: define{{.*}} void @testintmul()
void testintmul() {

  // CHECK:      [[T1:%.*]] = load i32, i32* @ij
  // CHECK-NEXT: [[T2:%.*]] = load i32, i32* @ik
  // CHECK-NEXT: [[T3:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 [[T1]], i32 [[T2]])
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T3]], 0
  // CHECK-NEXT: [[T5:%.*]] = extractvalue { i32, i1 } [[T3]], 1
  // CHECK:      call void @__ubsan_handle_mul_overflow
  ii = ij * ik;
}

// CHECK-LABEL: define{{.*}} void @testintpostinc()
void testintpostinc() {
  opaqueint(ii++);

  // CHECK:      [[T1:%.*]] = load i32, i32* @ii
  // CHECK-NEXT: [[T2:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i32, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T2]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
}

// CHECK-LABEL: define{{.*}} void @testintpreinc()
void testintpreinc() {
  opaqueint(++ii);

  // CHECK:      [[T1:%.*]] = load i32, i32* @ii
  // CHECK-NEXT: [[T2:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[T1]], i32 1)
  // CHECK-NEXT: [[T3:%.*]] = extractvalue { i32, i1 } [[T2]], 0
  // CHECK-NEXT: [[T4:%.*]] = extractvalue { i32, i1 } [[T2]], 1
  // CHECK:      call void @__ubsan_handle_add_overflow
}
