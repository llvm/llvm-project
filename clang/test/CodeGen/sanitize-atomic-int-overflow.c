// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -fsanitize=unsigned-integer-overflow %s -emit-llvm -o - | FileCheck %s

_Atomic(unsigned) atomic;

// CHECK-LABEL: define void @cmpd_assign
void cmpd_assign() {
  // CHECK: br label %[[LOOP_START:.*]]

  // CHECK: [[LOOP_START]]:
  // CHECK-NEXT: phi i32 {{.*}}, [ {{.*}}, %[[INCOMING_BLOCK:.*]] ]

  // CHECK: [[INCOMING_BLOCK]]:
  // CHECK-NEXT: cmpxchg
  // CHECK-NEXT: extractvalue
  // CHECK-NEXT: extractvalue
  // CHECK-NEXT: br i1 %8, label %{{.*}}, label %[[LOOP_START]]
  atomic += 1;
}

// CHECK-LABEL: define void @inc
void inc() {
  // CHECK: br label %[[LOOP_START:.*]]

  // CHECK: [[LOOP_START]]:
  // CHECK-NEXT: phi i32 {{.*}}, [ {{.*}}, %[[INCOMING_BLOCK:.*]] ]

  // CHECK: [[INCOMING_BLOCK]]:
  // CHECK-NEXT: cmpxchg
  // CHECK-NEXT: extractvalue
  // CHECK-NEXT: extractvalue
  // CHECK-NEXT: br i1 %8, label %{{.*}}, label %[[LOOP_START]]
  atomic++;
}
