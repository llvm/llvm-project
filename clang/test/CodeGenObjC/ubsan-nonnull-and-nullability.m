// REQUIRES: asserts
// RUN: %clang_cc1 -x objective-c -emit-llvm -triple x86_64-apple-macosx10.10.0 -fsanitize=nullability-return,returns-nonnull-attribute,nullability-arg,nonnull-attribute %s -o - -w | FileCheck %s

// If both the annotation and the attribute are present, prefer the attribute,
// since it actually affects IRGen.

// CHECK-LABEL: define nonnull i32* @f1
__attribute__((returns_nonnull)) int *_Nonnull f1(int *_Nonnull p) {
  // CHECK: entry:
  // CHECK-NEXT: [[ADDR:%.*]] = alloca i32*
  // CHECK-NEXT: store i32* [[P:%.*]], i32** [[ADDR]]
  // CHECK-NEXT: [[ARG:%.*]] = load i32*, i32** [[ADDR]]
  // CHECK-NEXT: [[ICMP:%.*]] = icmp ne i32* [[ARG]], null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], label %[[CONT:.+]], label %[[HANDLE:[^,]+]]
  // CHECK: [[HANDLE]]:
  // CHECK-NEXT:   call void @__ubsan_handle_nonnull_return_abort
  // CHECK-NEXT:   unreachable, !nosanitize
  // CHECK: [[CONT]]:
  // CHECK-NEXT:   ret i32*
  return p;
}

// CHECK-LABEL: define void @f2
void f2(int *_Nonnull __attribute__((nonnull)) p) {}

// CHECK-LABEL: define void @call_f2
void call_f2() {
  // CHECK: call void @__ubsan_handle_nonnull_arg_abort
  // CHECK-NOT: call void @__ubsan_handle_nonnull_arg_abort
  f2((void *)0);
}
