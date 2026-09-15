// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

template <const int N>
void template_foo() {
  int a = N + 5;
}

// CIR: %[[INIT:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!s32i>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[ADD:.*]] = cir.add nsw %[[CONST_1]], %[[CONST_2]] : !s32i
// CIR: cir.store{{.*}} %[[ADD]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[INIT:.*]] = alloca i32, align 4
// LLVM: store i32 6, ptr %[[INIT]], align 4

// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: store i32 6, ptr %[[INIT]], align 4

void foo() {
  template_foo<1>();
}

struct Point {
  int x;
  int y;
};

template <Point P>
void template_agg() {
  Point a = P;
}

// CIR: %[[A:.*]] = cir.alloca {{.*}}
// CIR: %[[GLOBAL:.*]] = cir.get_global @{{.*}}
// CIR: cir.copy %[[GLOBAL]] to %[[A]] : !cir.ptr<{{.*}}>
// CIR: cir.return

// LLVM: %[[A:.*]] = alloca %struct.Point
// LLVM: call void @llvm.memcpy.p0.p0.i64(ptr align {{[0-9]+}} %[[A]], ptr align {{[0-9]+}} @{{.*}}, i64 8, i1 false)
// LLVM: ret void

// OGCG: %[[A:.*]] = alloca %struct.Point
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align {{[0-9]+}} %[[A]], ptr align {{[0-9]+}} @{{.*}}, i64 8, i1 false)
// OGCG: ret void

void bar() {
  template_agg<Point{1, 2}>();
}
