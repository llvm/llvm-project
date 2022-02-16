// REQUIRES: asserts
// RUN: %clang_cc1 -x objective-c -emit-llvm -triple x86_64-apple-macosx10.10.0 -fsanitize=nullability-return,returns-nonnull-attribute,nullability-arg,nonnull-attribute %s -o - -w | FileCheck %s

// If both the annotation and the attribute are present, prefer the attribute,
// since it actually affects IRGen.

// CHECK-LABEL: define{{.*}} nonnull i32* @f1
__attribute__((returns_nonnull)) int *_Nonnull f1(int *_Nonnull p) {
  // CHECK: entry:
  // CHECK-NEXT: [[SLOC_PTR:%.*]] = alloca i8*
  // CHECK-NEXT: [[ADDR:%.*]] = alloca i32*
  // CHECK-NEXT: store i8* null, i8** [[SLOC_PTR]]
  // CHECK-NEXT: store i32* [[P:%.*]], i32** [[ADDR]]
  // CHECK-NEXT: store {{.*}} [[SLOC_PTR]]
  // CHECK-NEXT: [[ARG:%.*]] = load i32*, i32** [[ADDR]]
  // CHECK-NEXT: [[SLOC:%.*]] = load {{.*}} [[SLOC_PTR]]
  // CHECK-NEXT: [[SLOC_NONNULL:%.*]] = icmp ne i8* [[SLOC]], null
  // CHECK-NEXT: br i1 [[SLOC_NONNULL]], label %nullcheck
  // 
  // CHECK: nullcheck:
  // CHECK-NEXT: [[ICMP:%.*]] = icmp ne i32* [[ARG]], null, !nosanitize
  // CHECK-NEXT: br i1 [[ICMP]], label %[[CONT:.+]], label %[[HANDLE:[^,]+]]
  // CHECK: [[HANDLE]]:
  // CHECK:      call void @__ubsan_handle_nonnull_return
  // CHECK-NEXT:   unreachable, !nosanitize
  // CHECK: [[CONT]]:
  // CHECK-NEXT:   br label %no.nullcheck
  // CHECK: no.nullcheck:
  // CHECK-NEXT: ret i32* [[ARG]]
  return p;
}

// CHECK-LABEL: define{{.*}} void @f2
void f2(int *_Nonnull __attribute__((nonnull)) p) {}

// CHECK-LABEL: define{{.*}} void @call_f2
void call_f2(void) {
  // CHECK: call void @__ubsan_handle_nonnull_arg_abort
  // CHECK-NOT: call void @__ubsan_handle_nonnull_arg_abort
  f2((void *)0);
}

// If the return value isn't meant to be checked, make sure we don't check it.
// CHECK-LABEL: define{{.*}} i32* @f3
int *f3(int *p) {
  // CHECK-NOT: return.sloc
  // CHECK-NOT: call{{.*}}ubsan
  return p;
}

// Check for a valid "return" source location, even when there is no return
// statement, to avoid accidentally calling the runtime.

// CHECK-LABEL: define{{.*}} nonnull i32* @f4
__attribute__((returns_nonnull)) int *f4(void) {
  // CHECK: store i8* null, i8** [[SLOC_PTR:%.*]]
  // CHECK: [[SLOC:%.*]] = load {{.*}} [[SLOC_PTR]]
  // CHECK: [[SLOC_NONNULL:%.*]] = icmp ne i8* [[SLOC]], null
  // CHECK: br i1 [[SLOC_NONNULL]], label %nullcheck
  // CHECK: nullcheck:
}
