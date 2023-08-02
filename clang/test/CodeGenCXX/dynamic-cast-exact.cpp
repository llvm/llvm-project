// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -std=c++11 -o - -O1 -disable-llvm-passes | FileCheck %s --implicit-check-not='call {{.*}} @__dynamic_cast'
struct Offset { virtual ~Offset(); };
struct A { virtual ~A(); };
struct B final : Offset, A { };

struct C { virtual ~C(); int c; };
struct D : A { int d; };
struct E : A { int e; };
struct F : virtual A { int f; };
struct G : virtual A { int g; };
struct H final : C, D, E, F, G { int h; };

// CHECK-LABEL: @_Z7inexactP1A
C *inexact(A *a) {
  // CHECK: call {{.*}} @__dynamic_cast
  return dynamic_cast<C*>(a);
}

// CHECK-LABEL: @_Z12exact_singleP1A
B *exact_single(A *a) {
  // CHECK: %[[PTR_NULL:.*]] = icmp eq ptr %[[PTR:.*]], null
  // CHECK: br i1 %[[PTR_NULL]], label %[[LABEL_FAILED:.*]], label %[[LABEL_NOTNULL:.*]]

  // CHECK: [[LABEL_NOTNULL]]:
  // CHECK: %[[VPTR:.*]] = load ptr, ptr %[[PTR]]
  // CHECK: %[[MATCH:.*]] = icmp eq ptr %[[VPTR]], getelementptr inbounds ({ [4 x ptr], [4 x ptr] }, ptr @_ZTV1B, i32 0, inrange i32 1, i32 2)
  // CHECK: %[[RESULT:.*]] = getelementptr inbounds i8, ptr %[[PTR]], i64 -8
  // CHECK: br i1 %[[MATCH]], label %[[LABEL_END:.*]], label %[[LABEL_FAILED]]

  // CHECK: [[LABEL_FAILED]]:
  // CHECK: br label %[[LABEL_END]]

  // CHECK: [[LABEL_END]]:
  // CHECK: phi ptr [ %[[RESULT]], %[[LABEL_NOTNULL]] ], [ null, %[[LABEL_FAILED]] ]
  return dynamic_cast<B*>(a);
}

// CHECK-LABEL: @_Z9exact_refR1A
B &exact_ref(A &a) {
  // CHECK: %[[PTR_NULL:.*]] = icmp eq ptr %[[PTR:.*]], null
  // CHECK: br i1 %[[PTR_NULL]], label %[[LABEL_FAILED:.*]], label %[[LABEL_NOTNULL:.*]]

  // CHECK: [[LABEL_NOTNULL]]:
  // CHECK: %[[VPTR:.*]] = load ptr, ptr %[[PTR]]
  // CHECK: %[[MATCH:.*]] = icmp eq ptr %[[VPTR]], getelementptr inbounds ({ [4 x ptr], [4 x ptr] }, ptr @_ZTV1B, i32 0, inrange i32 1, i32 2)
  // CHECK: %[[RESULT:.*]] = getelementptr inbounds i8, ptr %[[PTR]], i64 -8
  // CHECK: br i1 %[[MATCH]], label %[[LABEL_END:.*]], label %[[LABEL_FAILED]]

  // CHECK: [[LABEL_FAILED]]:
  // CHECK: call {{.*}} @__cxa_bad_cast
  // CHECK: unreachable

  // CHECK: [[LABEL_END]]:
  // CHECK: ret ptr %[[RESULT]]
  return dynamic_cast<B&>(a);
}

// CHECK-LABEL: @_Z11exact_multiP1A
H *exact_multi(A *a) {
  // CHECK: %[[PTR_NULL:.*]] = icmp eq ptr %[[PTR:.*]], null
  // CHECK: br i1 %[[PTR_NULL]], label %[[LABEL_FAILED:.*]], label %[[LABEL_NOTNULL:.*]]

  // CHECK: [[LABEL_NOTNULL]]:
  // CHECK: %[[VPTR:.*]] = load ptr, ptr %[[PTR]]
  // CHECK: %[[OFFSET_TO_TOP_SLOT:.*]] = getelementptr inbounds i64, ptr %[[VPTR]], i64 -2
  // CHECK: %[[OFFSET_TO_TOP:.*]] = load i64, ptr %[[OFFSET_TO_TOP_SLOT]]
  // CHECK: %[[RESULT:.*]] = getelementptr inbounds i8, ptr %[[PTR]], i64 %[[OFFSET_TO_TOP]]
  // CHECK: %[[DERIVED_VPTR:.*]] = load ptr, ptr %[[RESULT]]
  // CHECK: %[[MATCH:.*]] = icmp eq ptr %[[DERIVED_VPTR]], getelementptr inbounds ({ [5 x ptr], [4 x ptr], [4 x ptr], [6 x ptr], [6 x ptr] }, ptr @_ZTV1H, i32 0, inrange i32 0, i32 3)
  // CHECK: br i1 %[[MATCH]], label %[[LABEL_END:.*]], label %[[LABEL_FAILED]]

  // CHECK: [[LABEL_FAILED]]:
  // CHECK: br label %[[LABEL_END]]

  // CHECK: [[LABEL_END]]:
  // CHECK: phi ptr [ %[[RESULT]], %[[LABEL_NOTNULL]] ], [ null, %[[LABEL_FAILED]] ]
  return dynamic_cast<H*>(a);
}

namespace GH64088 {
  // Ensure we mark the B vtable as used here, because we're going to emit a
  // reference to it.
  // CHECK: define {{.*}} @_ZN7GH640881BD0
  struct A { virtual ~A(); };
  struct B final : A { virtual ~B() = default; };
  B *cast(A *p) { return dynamic_cast<B*>(p); }
}
