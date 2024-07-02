// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=gnu11 -verify -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

union U1 {
  int x;
  char y[16];
};
// CHECK: %union.U1 = type { i32, [12 x i8] }

struct S1 {
  int x;
  union U1 y;
};
// CHECK: %struct.S1 = type { i32, %union.U1 }

union U2 {
  int x;
  char y[16];
} __attribute__((__aligned__(32)));
// CHECK: %union.U2 = type { i32, [28 x i8] }

struct S2 {
  int x;
  long long y;
  char z[8];
} __attribute__((__aligned__(32)));
// CHECK: %struct.S2 = type { i32, i64, [8 x i8], [8 x i8] }

union U1 global_u1 = {};
// CHECK: @global_u1 ={{.*}} global %union.U1 zeroinitializer, align 4

union U1 global_u2 = {3};
// CHECK: @global_u2 ={{.*}} global %union.U1 { i32 3, [12 x i8] zeroinitializer }, align 4

struct S1 global_s1 = {};
// CHECK: @global_s1 ={{.*}} global %struct.S1 zeroinitializer, align 4

struct S1 global_s2 = {
    .x = 3,
};
// CHECK: @global_s2 ={{.*}} global %struct.S1 { i32 3, %union.U1 zeroinitializer }, align 4

struct S1 global_s3 = {.x = 3, .y = {.x = 6}};
// CHECK: @global_s3 ={{.*}} global %struct.S1 { i32 3, %union.U1 { i32 6, [12 x i8] zeroinitializer } }, align 4

union U2 global_u3 = {};
// CHECK: @global_u3 ={{.*}} global %union.U2 zeroinitializer, align 32

struct S2 global_s4 = {};
// CHECK: @global_s4 ={{.*}} global { i32, [4 x i8], i64, [8 x i8], [8 x i8] } zeroinitializer, align 32

struct S2 global_s5 = {.x = 1};
// CHECK: @global_s5 ={{.*}}global { i32, [4 x i8], i64, [8 x i8], [8 x i8] } { i32 1, [4 x i8] zeroinitializer, i64 0, [8 x i8] zeroinitializer, [8 x i8] zeroinitializer }, align 32

// CHECK: @test2.a ={{.*}} global %union.U1 zeroinitializer, align 4
// CHECK: @__const.test3.a  ={{.*}} constant %union.U1 { i32 3, [12 x i8] zeroinitializer }, align 4
// CHECK: @test4.a ={{.*}} global %union.U1 { i32 3, [12 x i8] zeroinitializer }, align 4
// CHECK: @test6.s ={{.*}} global %struct.S1 zeroinitializer, align 4
// CHECK: @__const.test7.s ={{.*}} constant %struct.S1 { i32 3, %union.U1 zeroinitializer }, align 4
// CHECK: @test8.s ={{.*}} global %struct.S1 { i32 3, %union.U1 zeroinitializer }, align 4
// CHECK: @__const.test9.s ={{.*}} constant %struct.S1 { i32 3, %union.U1 { i32 6, [12 x i8] zeroinitializer } }, align 4
// CHECK: @test10.s ={{.*}} global %struct.S1 { i32 3, %union.U1 { i32 6, [12 x i8] zeroinitializer } }, align 4
// CHECK: @test12.a ={{.*}} global %union.U2 zeroinitializer, align 32
// CHECK: @test14.s ={{.*}} global { i32, [4 x i8], i64, [8 x i8], [8 x i8] } zeroinitializer, align 32
// CHECK: @__const.test15.s ={{.*}} constant { i32, [4 x i8], i64, [8 x i8], [8 x i8] } { i32 1, [4 x i8] zeroinitializer, i64 0, [8 x i8] zeroinitializer, [8 x i8] zeroinitializer }, align 32
// CHECK: @test16.s = internal global { i32, [4 x i8], i64, [8 x i8], [8 x i8] } { i32 1, [4 x i8] zeroinitializer, i64 0, [8 x i8] zeroinitializer, [8 x i8] zeroinitializer }, align 32

// Test empty initializer for union.
void test1() {
  union U1 a = {};
  // CHECK-LABEL: @test1()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[A:.+]] = alloca %union.U1, align 4
  // CHECK-NEXT: call void @llvm.memset.p0.i64({{.*}}%[[A]], i8 0, i64 16, i1 false)
}

// Test empty initializer for union. Use static variable.
void test2() {
  static union U1 a = {};
  // CHECK-LABEL: @test2()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: ret void
}

// Test only initializing a small field for union.
void test3() {
  union U1 a = {3};
  // CHECK-LABEL: @test3()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[A:.+]] = alloca %union.U1
  // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64({{.*}}%[[A]], {{.*}}@__const.test3.a, i64 16, i1 false)
}

// Test only initializing a small field for union. Use static variable.
void test4() {
  static union U1 a = {3};
  // CHECK-LABEL: @test4()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: ret void
}

// Test union in struct. Use empty initializer for the struct.
void test5() {
  struct S1 s = {};
  // CHECK-LABEL: @test5()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[S:.+]] = alloca %struct.S1
  // CHECK-NEXT: call void @llvm.memset.p0.i64({{.*}}%[[S]], i8 0, i64 20, i1 false)
}

// Test union in struct. Use empty initializer for the struct. Use static variable.
void test6() {
  static struct S1 s = {};
  // CHECK-LABEL: @test6()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: ret void
}

// Test union in struct. Initialize other fields of the struct.
void test7() {
  struct S1 s = {
      .x = 3,
  };
  // CHECK-LABEL: @test7()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[S:.+]] = alloca %struct.S1
  // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64({{.*}}%[[S]], {{.*}}@__const.test7.s, i64 20, i1 false)
}

// Test union in struct. Initialize other fields of the struct. Use static variable.
void test8() {
  static struct S1 s = {
      .x = 3,
  };
  // CHECK-LABEL: @test8()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: ret void
}

// Test union in struct. Initialize a small field for union.
void test9() {
  struct S1 s = {.x = 3,
                .y = {
                    .x = 6,
                }};
  // CHECK-LABEL: @test9()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[S:.+]] = alloca %struct.S1
  // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64({{.*}}%[[S]], {{.*}}@__const.test9.s, i64 20, i1 false)
}

// Test union in struct. Initialize a small field for union. Use static variable.
void test10() {
  static struct S1 s = {.x = 3,
                       .y = {
                           .x = 6,
                       }};
  // CHECK-LABEL: @test10()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: ret void
}

// Test empty initializer for union with padding.
void test11() {
  union U2 a = {};
  // CHECK-LABEL: @test11()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[A:.+]] = alloca %union.U2, align 32
  // CHECK-NEXT: call void @llvm.memset.p0.i64({{.*}}%[[A]], i8 0, i64 32, i1 false)
}

// Test empty initializer for union with padding. Use static variable.
void test12() {
  static union U2 a = {};
  // CHECK-LABEL: @test12()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: ret void
}

// Test empty initializer for struct with padding.
void test13() {
  struct S2 s = {};
  // CHECK-LABEL: @test13()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[S:.+]] = alloca %struct.S2, align 32
  // CHECK-NEXT: call void @llvm.memset.p0.i64({{.*}}%[[S]], i8 0, i64 32, i1 false)
}

// Test empty initializer for struct with padding. Use static variable.
void test14() {
  static struct S2 s = {};
  // CHECK-LABEL: @test14()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: ret void
}

// Test partial initialization for struct with padding.
void test15() {
  struct S2 s = {.x = 1};
  // CHECK-LABEL: @test15()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[S:.+]] = alloca %struct.S2, align 32
  // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64({{.*}}%[[S]], {{.*}}@__const.test15.s, i64 32, i1 false)
}

// Test partial initialization for struct with padding. Use static variable.
void test16() {
  static struct S2 s = {.x = 1};
  // CHECK-LABEL: @test16()
  // CHECK-NEXT: entry:
  // CHECK-NEXT: ret void
}
