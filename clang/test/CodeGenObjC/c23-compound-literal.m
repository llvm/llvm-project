// RUN: %clang_cc1 -std=c23 -triple x86_64-apple-macosx10.15 -emit-llvm -fobjc-arc -disable-llvm-passes -o - %s | FileCheck %s

typedef struct {
  id a;
} S;
int f1(const S *);

// CHECK-LABEL: define i32 @f2(
// CHECK: %.compoundliteral = alloca %struct.S, align 8
// CHECK: store ptr null, ptr %{{.*}}, align 8
// CHECK: %[[CALL:.*]] = call i32 @f1(ptr noundef %.compoundliteral)
// CHECK-NEXT: zext i32 %[[CALL]] to i64
// CHECK: %[[RESULT:.*]] = load i32, ptr %{{.*}}, align 4
// CHECK-NEXT: call void @__destructor_8_s0(ptr %.compoundliteral)
// CHECK-NEXT: ret i32 %[[RESULT]]
// CHECK-NEXT: }
int f2(int a[f1(&(constexpr S){.a = 0})]) {
  return a[0];
}
