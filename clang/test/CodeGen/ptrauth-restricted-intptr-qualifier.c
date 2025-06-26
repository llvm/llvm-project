// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-intrinsics -emit-llvm %s -O0 -o - | FileCheck %s

#define __ptrauth(...) __ptrauth(__VA_ARGS__)

__INTPTR_TYPE__ __ptrauth(1, 0, 56) g1 = 0;
// CHECK: @g1 = global i64 0
__INTPTR_TYPE__ __ptrauth(1, 1, 1272) g2 = 0;
// CHECK: @g2 = global i64 0
extern __UINTPTR_TYPE__ test_int;
__UINTPTR_TYPE__ __ptrauth(3, 1, 23) g3 = (__UINTPTR_TYPE__)&test_int;
// CHECK: @test_int = external global i64
// CHECK: @g3 = global i64 ptrtoint (ptr ptrauth (ptr @test_int, i32 3, i64 23, ptr @g3) to i64)

__INTPTR_TYPE__ __ptrauth(1, 1, 712) ga[3] = {0,0,(__UINTPTR_TYPE__)&test_int};

// CHECK: @ga = global [3 x i64] [i64 0, i64 0, i64 ptrtoint (ptr ptrauth (ptr @test_int, i32 1, i64 712, ptr getelementptr inbounds ([3 x i64], ptr @ga, i32 0, i32 2)) to i64)]

struct A {
  __INTPTR_TYPE__ __ptrauth(1, 0, 431) f0;
  __INTPTR_TYPE__ __ptrauth(1, 0, 9182) f1;
  __INTPTR_TYPE__ __ptrauth(1, 0, 783) f2;
};

struct A gs1 = {0, 0, (__UINTPTR_TYPE__)&test_int};
// CHECK: @gs1 = global %struct.A { i64 0, i64 0, i64 ptrtoint (ptr ptrauth (ptr @test_int, i32 1, i64 783) to i64) }

struct B {
  __INTPTR_TYPE__ __ptrauth(1, 1, 1276) f0;
  __INTPTR_TYPE__ __ptrauth(1, 1, 23674) f1;
  __INTPTR_TYPE__ __ptrauth(1, 1, 163) f2;
};

struct B gs2 = {0, 0, (__UINTPTR_TYPE__)&test_int};
// CHECK: @gs2 = global %struct.B { i64 0, i64 0, i64 ptrtoint (ptr ptrauth (ptr @test_int, i32 1, i64 163, ptr getelementptr inbounds (%struct.B, ptr @gs2, i32 0, i32 2)) to i64) }

// CHECK-LABEL: i64 @test_read_globals
__INTPTR_TYPE__ test_read_globals() {
  __INTPTR_TYPE__ result = g1 + g2 + g3;
  // CHECK: [[A:%.*]] = load i64, ptr @g1
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[A]], i32 1, i64 56)
  // CHECK: [[B:%.*]] = load i64, ptr @g2
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @g2 to i64), i64 1272)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[B]], i32 1, i64 [[BLENDED]])
  // CHECK: [[VALUE:%.*]] = load i64, ptr @g3
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @g3 to i64), i64 23)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VALUE]], i32 3, i64 [[BLENDED]])

  for (int i = 0; i < 3; i++) {
    result += ga[i];
  }
  // CHECK: for.cond:
  // CHECK: [[TEMP:%.*]] = load i32, ptr [[IDX_ADDR:%.*]]

  // CHECK: for.body:
  // CHECK: [[IDX:%.*]] = load i32, ptr [[IDX_ADDR]]
  // CHECK: [[IDXPROM:%.*]] = sext i32 [[IDX]] to i64
  // CHECK: [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x i64], ptr @ga, i64 0, i64 [[IDXPROM]]
  // CHECK: [[VALUE:%.*]] = load i64, ptr [[ARRAYIDX]]
  // CHECK: [[CASTIDX:%.*]] = ptrtoint ptr [[ARRAYIDX]] to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CASTIDX]], i64 712)
  // CHECK: resign.nonnull6:
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VALUE]], i32 1, i64 [[BLENDED]])
  // CHECK: resign.cont7

  result += gs1.f0 + gs1.f1 + gs1.f2;
  // CHECK: resign.cont10:
  // CHECK: [[ADDR:%.*]] = load i64, ptr getelementptr inbounds nuw (%struct.A, ptr @gs1, i32 0, i32 1
  // CHECK: resign.nonnull11:
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[ADDR]], i32 1, i64 9182)
  // CHECK: resign.cont12:
  // CHECK: [[ADDR:%.*]] = load i64, ptr getelementptr inbounds nuw (%struct.A, ptr @gs1, i32 0, i32 2)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[ADDR]], i32 1, i64 783)
  result += gs2.f0 + gs2.f1 + gs2.f2;
  // CHECK: [[ADDR:%.*]] = load i64, ptr @gs2
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr @gs2 to i64), i64 1276)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[ADDR]], i32 1, i64 [[BLENDED]])
  // CHECK: [[ADDR:%.*]] = load i64, ptr getelementptr inbounds nuw (%struct.B, ptr @gs2, i32 0, i32 1)
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr getelementptr inbounds nuw (%struct.B, ptr @gs2, i32 0, i32 1) to i64), i64 23674)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[ADDR]], i32 1, i64 [[BLENDED]])
  // CHECK: [[ADDR:%.*]] = load i64, ptr getelementptr inbounds nuw (%struct.B, ptr @gs2, i32 0, i32 2)
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr getelementptr inbounds nuw (%struct.B, ptr @gs2, i32 0, i32 2) to i64), i64 163)

  return result;
}

// CHECK-LABEL: void @test_write_globals
void test_write_globals(int i, __INTPTR_TYPE__ j) {
  g1 = i;
  g2 = j;
  g3 = 0;
  ga[0] = i;
  ga[1] = j;
  ga[2] = 0;
  gs1.f0 = i;
  gs1.f1 = j;
  gs1.f2 = 0;
  gs2.f0 = i;
  gs2.f1 = j;
  gs2.f2 = 0;
}

// CHECK-LABEL: define void @test_set_A
void test_set_A(struct A *a, __INTPTR_TYPE__ x, int y) {
  a->f0 = x;
  // CHECK: [[XADDR:%.*]] = load i64, ptr %x.addr
  // CHECK: [[SIGNED_X:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[XADDR]], i32 1, i64 431)
  a->f1 = y;
  // CHECK: [[Y:%.*]] = load i32, ptr %y.addr
  // CHECK: [[CONV:%.*]] = sext i32 [[Y]] to i64
  // CHECK: [[SIGNED:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[CONV]], i32 1, i64 9182)
  a->f2 = 0;
  // CHECK: [[A:%.*]] = load ptr, ptr %a.addr
  // CHECK: [[F2:%.*]] = getelementptr inbounds nuw %struct.A, ptr [[A]], i32 0, i32 2
  // CHECK: store i64 0, ptr [[F2]]
}

// CHECK-LABEL: define void @test_set_B
void test_set_B(struct B *b, __INTPTR_TYPE__ x, int y) {
  b->f0 = x;
  // CHECK: [[X:%.*]] = load i64, ptr %x.addr
  // CHECK: [[F0_ADDR:%.*]] = ptrtoint ptr %f0 to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[F0_ADDR]], i64 1276)
  // CHECK: [[SIGNED:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[X]], i32 1, i64 [[BLENDED]])
  b->f1 = y;
  // CHECK: [[B:%.*]] = load ptr, ptr %b.addr
  // CHECK: [[F1_ADDR:%.*]] = getelementptr inbounds nuw %struct.B, ptr [[B]], i32 0, i32 1
  // CHECK: [[Y:%.*]] = load i32, ptr %y.addr, align 4
  // CHECK: [[CONV:%.*]] = sext i32 [[Y]] to i64
  // CHECK: [[CAST_ADDR:%.*]] = ptrtoint ptr [[F1_ADDR]] to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_ADDR]], i64 23674)
  // CHECK: [[SIGNED:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[CONV]], i32 1, i64 [[BLENDED]])
  b->f2 = 0;
  // CHECK: [[B:%.*]] = load ptr, ptr %b.addr
  // CHECK: [[F2_ADDR:%.*]] = getelementptr inbounds nuw %struct.B, ptr [[B]], i32 0, i32 2
  // CHECK: store i64 0, ptr [[F2_ADDR]]
}

// CHECK-LABEL: define i64 @test_get_A
__INTPTR_TYPE__ test_get_A(struct A *a) {
  return a->f0 + a->f1 + a->f2;
  // CHECK: [[A:%.*]] = load ptr, ptr %a.addr
  // CHECK: [[F0_ADDR:%.*]] = getelementptr inbounds nuw %struct.A, ptr [[A]], i32 0, i32 0
  // CHECK: [[F0:%.*]] = load i64, ptr [[F0_ADDR]]
  // CHECK: [[AUTH:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[F0]], i32 1, i64 431)
  // CHECK: [[A:%.*]] = load ptr, ptr %a.addr
  // CHECK: [[F1_ADDR:%.*]] = getelementptr inbounds nuw %struct.A, ptr [[A]], i32 0, i32 1
  // CHECK: [[F1:%.*]] = load i64, ptr [[F1_ADDR]]
  // CHECK: [[AUTH:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[F1]], i32 1, i64 9182)
  // CHECK: [[A:%.*]] = load ptr, ptr %a.addr
  // CHECK: [[F2_ADDR:%.*]] = getelementptr inbounds nuw %struct.A, ptr [[A]], i32 0, i32 2
  // CHECK: [[F2:%.*]] = load i64, ptr [[F2_ADDR]]
  // CHECK: [[AUTH:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[F2]], i32 1, i64 783)
}

// CHECK-LABEL: define i64 @test_get_B
__INTPTR_TYPE__ test_get_B(struct B *b) {
  return b->f0 + b->f1 + b->f2;
  // CHECK: [[B:%.*]] = load ptr, ptr %b.addr
  // CHECK: [[F0:%.*]] = getelementptr inbounds nuw %struct.B, ptr [[B]], i32 0, i32 0
  // CHECK: [[VALUE:%.*]] = load i64, ptr [[F0]]
  // CHECK: [[CASTF0:%.*]] = ptrtoint ptr %f0 to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CASTF0]], i64 1276)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VALUE]], i32 1, i64 [[BLENDED]])
  // CHECK: [[B:%.*]] = load ptr, ptr %b.addr
  // CHECK: [[ADDR:%.*]] = getelementptr inbounds nuw %struct.B, ptr [[B]], i32 0, i32 1
  // CHECK: [[VALUE:%.*]] = load i64, ptr [[ADDR]]
  // CHECK: [[CAST_ADDR:%.*]] = ptrtoint ptr [[ADDR]] to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_ADDR]], i64 23674)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VALUE]], i32 1, i64 [[BLENDED]])
  // CHECK: [[B:%.*]] = load ptr, ptr %b.addr
  // CHECK: [[ADDR:%.*]] = getelementptr inbounds nuw %struct.B, ptr [[B]], i32 0, i32 2
  // CHECK: [[VALUE:%.*]] = load i64, ptr [[ADDR]]
  // CHECK: [[CAST_ADDR:%.*]] = ptrtoint ptr [[ADDR]] to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_ADDR]], i64 163)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VALUE]], i32 1, i64 [[BLENDED]])
}

// CHECK-LABEL: define void @test_resign
void test_resign(struct A* a, const struct B *b) {
  a->f0 = b->f0;
  // CHECK: [[A:%.*]] = load ptr, ptr %a.addr, align 8
  // CHECK: [[F0:%.*]] = getelementptr inbounds nuw %struct.A, ptr [[A]], i32 0, i32 0
  // CHECK: [[B:%.*]] = load ptr, ptr %b.addr, align 8
  // CHECK: [[F01:%.*]] = getelementptr inbounds nuw %struct.B, ptr [[B]], i32 0, i32 0
  // CHECK: [[F01VALUE:%.*]] = load i64, ptr [[F01]]
  // CHECK: [[CASTF01:%.*]] = ptrtoint ptr %f01 to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CASTF01]], i64 1276)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[F01VALUE]], i32 1, i64 [[BLENDED]], i32 1, i64 431)
}

// CHECK-LABEL: define i64 @other_test
__INTPTR_TYPE__ other_test(__INTPTR_TYPE__ i) {
  __INTPTR_TYPE__ __ptrauth(1, 1, 42) j = 0;
  // CHECK: [[J_ADDR:%.*]] = ptrtoint ptr %j to i64
  // CHECK: store i64 0, ptr %j
  __INTPTR_TYPE__ __ptrauth(1, 1, 43) k = 1234;
  // CHECK: [[ADDR:%.*]] = ptrtoint ptr %k to i64
  // CHECK: [[JBLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[ADDR]], i64 43)
  // CHECK: [[SIGNED:%.*]] = call i64 @llvm.ptrauth.sign(i64 1234, i32 1, i64 [[JBLENDED]])
  __INTPTR_TYPE__ __ptrauth(1, 1, 44) l = i;
  // CHECK: [[I:%.*]] = load i64, ptr %i.addr
  // CHECK: [[ADDR:%.*]] = ptrtoint ptr %l to i64
  // CHECK: [[LBLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[ADDR]], i64 44)
  // CHECK: [[SIGNED:%.*]] = call i64 @llvm.ptrauth.sign(i64 [[I]], i32 1, i64 [[LBLENDED]])
  asm volatile ("" ::: "memory");
  return j + k + l;
  // CHECK: [[VALUE:%.*]] = load i64, ptr %j
  // CHECK: [[CAST_ADDR:%.*]] = ptrtoint ptr %j to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CAST_ADDR]], i64 42)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VALUE]], i32 1, i64 [[BLENDED]])
  // CHECK: [[VALUE:%.*]] = load i64, ptr %k
  // CHECK: [[CASTK:%.*]] = ptrtoint ptr %k to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CASTK]], i64 43)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VALUE]], i32 1, i64 [[BLENDED]])
  // CHECK: [[VALUE:%.*]] = load i64, ptr %l
  // CHECK: [[CASTL:%.*]] = ptrtoint ptr %l to i64
  // CHECK: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[CASTL]], i64 44)
  // CHECK: [[AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VALUE]], i32 1, i64 [[BLENDED]])
}
