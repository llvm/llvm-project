// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c2x -verify -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

struct S { int x, y; };
struct T {
  int i;
  struct S s;
};

// CHECK: @[[CONST_T1:.+]] = private unnamed_addr constant %struct.T { i32 1, %struct.S zeroinitializer }
// CHECK: @[[CONST_T2:.+]] = private unnamed_addr constant %struct.T { i32 1, %struct.S { i32 2, i32 0 } }

void test_struct() {
  struct S s = {};
  // CHECK: define {{.*}} void @test_struct
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[S:.+]] = alloca %struct.S
  // CHECK-NEXT: call void @llvm.memset.p0.i64({{.*}}%[[S]], i8 0, i64 8, i1 false)
}

void test_var() {
  int i = {};
  // CHECK: define {{.*}} void @test_var
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[I:.+]] = alloca i32
  // CHECK-NEXT: store i32 0, ptr %[[I]]
}

void test_simple_compound_literal() {
  int j = (int){};
  // CHECK: define {{.*}} void @test_simple_compound_literal
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[J:.+]] = alloca i32
  // CHECK-NEXT: %[[COMPOUND:.+]] = alloca i32
  // CHECK-NEXT: store i32 0, ptr %[[COMPOUND]]
  // CHECK-NEXT: %[[MEM:.+]] = load i32, ptr %[[COMPOUND]]
  // CHECK-NEXT: store i32 %[[MEM]], ptr %[[J]]
}

void test_zero_size_array() {
  int unknown_size[] = {};
  // CHECK: define {{.*}} void @test_zero_size_array
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[UNKNOWN:.+]] = alloca [0 x i32]
}

void test_vla() {
  int num_elts = 12;
  int vla[num_elts] = {};
  // CHECK: define {{.*}} void @test_vla
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[NUM_ELTS_PTR:.+]] = alloca i32
  // CHECK: %[[VLA_EXPR:.+]] = alloca i64
  // CHECK-NEXT: store i32 12, ptr %[[NUM_ELTS_PTR]]
  // CHECK-NEXT: %[[NUM_ELTS:.+]] = load i32, ptr %[[NUM_ELTS_PTR]]
  // CHECK-NEXT: %[[NUM_ELTS_EXT:.+]] = zext i32 %[[NUM_ELTS]] to i64
  // CHECK: %[[VLA:.+]] = alloca i32, i64 %[[NUM_ELTS_EXT]]
  // CHECK-NEXT: store i64 %[[NUM_ELTS_EXT]], ptr %[[VLA_EXPR]]
  // CHECK-NEXT: %[[BYTES_TO_COPY:.+]] = mul nuw i64 %[[NUM_ELTS_EXT]], 4
  // CHECK-NEXT: call void @llvm.memset.p0.i64(ptr {{.*}} %[[VLA]], i8 0, i64 %[[BYTES_TO_COPY]], i1 false)
}

void test_zero_size_vla() {
  int num_elts = 0;
  int vla[num_elts] = {};
  // CHECK: define {{.*}} void @test_zero_size_vla
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[NUM_ELTS_PTR:.+]] = alloca i32
  // CHECK: %[[VLA_EXPR:.+]] = alloca i64
  // CHECK-NEXT: store i32 0, ptr %[[NUM_ELTS_PTR]]
  // CHECK-NEXT: %[[NUM_ELTS:.+]] = load i32, ptr %[[NUM_ELTS_PTR]]
  // CHECK-NEXT: %[[NUM_ELTS_EXT:.+]] = zext i32 %[[NUM_ELTS]] to i64
  // CHECK: %[[VLA:.+]] = alloca i32, i64 %[[NUM_ELTS_EXT]]
  // CHECK-NEXT: store i64 %[[NUM_ELTS_EXT]], ptr %[[VLA_EXPR]]
  // CHECK-NEXT: %[[BYTES_TO_COPY:.+]] = mul nuw i64 %[[NUM_ELTS_EXT]], 4
  // CHECK-NEXT: call void @llvm.memset.p0.i64(ptr {{.*}} %[[VLA]], i8 0, i64 %[[BYTES_TO_COPY]], i1 false)
}

void test_compound_literal_vla() {
  int num_elts = 12;
  int *compound_literal_vla = (int[num_elts]){};
  // CHECK: define {{.*}} void @test_compound_literal_vla
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[NUM_ELTS_PTR:.+]] = alloca i32
  // CHECK-NEXT: %[[COMP_LIT_VLA:.+]] = alloca ptr
  // CHECK-NEXT: %[[COMP_LIT:.+]] = alloca i32
  // CHECK-NEXT: store i32 12, ptr %[[NUM_ELTS_PTR]]
  // CHECK-NEXT: %[[NUM_ELTS:.+]] = load i32, ptr %[[NUM_ELTS_PTR]]
  // CHECK-NEXT: %[[NUM_ELTS_EXT:.+]] = zext i32 %[[NUM_ELTS]] to i64
  // CHECK-NEXT: %[[BYTES_TO_COPY:.+]] = mul nuw i64 %[[NUM_ELTS_EXT]], 4
  // CHECK-NEXT: call void @llvm.memset.p0.i64(ptr {{.*}} %[[COMP_LIT]], i8 0, i64 %[[BYTES_TO_COPY]], i1 false)
  // CHECK-NEXT: store ptr %[[COMP_LIT]], ptr %[[COMP_LIT_VLA]]
}

void test_nested_structs() {
  struct T t1 = { 1, {} };
  struct T t2 = { 1, { 2, {} } };
  struct T t3 = { (int){}, {} };
  // CHECK: define {{.*}} void @test_nested_structs
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[T1:.+]] = alloca %struct.T
  // CHECK-NEXT: %[[T2:.+]] = alloca %struct.T
  // CHECK-NEXT: %[[T3:.+]] = alloca %struct.T
  // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[T1]], ptr {{.*}} @[[CONST_T1]], i64 12, i1 false)
  // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}} %[[T2]], ptr {{.*}} @[[CONST_T2]], i64 12, i1 false)
  // CHECK-NEXT: call void @llvm.memset.p0.i64(ptr {{.*}} %[[T3]], i8 0, i64 12, i1 false)
}

void test_vla_of_nested_structs(int num_elts) {
  struct T t3[num_elts] = {};
  // CHECK: define {{.*}} void @test_vla_of_nested_structs(i32 noundef %[[NUM_ELTS_PARAM:.+]])
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[NUM_ELTS_PTR:.+]] = alloca i32
  // CHECK: %[[VLA_EXPR:.+]] = alloca i64
  // CHECK-NEXT: store i32 %[[NUM_ELTS_PARAM]], ptr %[[NUM_ELTS_PTR]]
  // CHECK-NEXT: %[[NUM_ELTS_LOCAL:.+]] = load i32, ptr %[[NUM_ELTS_PTR]]
  // CHECK-NEXT: %[[NUM_ELTS_EXT:.+]] = zext i32 %[[NUM_ELTS_LOCAL]] to i64
  // CHECK: %[[VLA:.+]] = alloca %struct.T, i64 %[[NUM_ELTS_EXT]]
  // CHECK-NEXT: store i64 %[[NUM_ELTS_EXT]], ptr %[[VLA_EXPR]]
  // CHECK-NEXT: %[[COPY_BYTES:.+]] = mul nuw i64 %[[NUM_ELTS_EXT]], 12
  // CHECK-NEXT: call void @llvm.memset.p0.i64(ptr {{.*}} %[[VLA]], i8 0, i64 %[[COPY_BYTES]], i1 false)
}
