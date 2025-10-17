// Simple functions
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o -  | FileCheck %s

void empty() { }
// CHECK: define{{.*}} void @_Z5emptyv()
// CHECK:   ret void

void voidret() { return; }
// CHECK: define{{.*}} void @_Z7voidretv()
// CHECK:   ret void

int intfunc() { return 42; }
// CHECK: define{{.*}} i32 @_Z7intfuncv()
// CHECK:   %[[RV:.*]] = alloca i32, i64 1, align 4
// CHECK:   store i32 42, ptr %[[RV]], align 4
// CHECK:   %[[R:.*]] = load i32, ptr %[[RV]], align 4
// CHECK:   ret i32 %[[R]]

int scopes() {
  {
    {
      return 99;
    }
  }
}
// CHECK: define{{.*}} i32 @_Z6scopesv(){{.*}} {
// CHECK:   %[[RV:.*]] = alloca i32, i64 1, align 4
// CHECK:   br label %[[LABEL1:.*]]
// CHECK: [[LABEL1]]:
// CHECK:   br label %[[LABEL2:.*]]
// CHECK: [[LABEL2]]:
// CHECK:   store i32 99, ptr %[[RV]], align 4
// CHECK:   %[[R:.*]] = load i32, ptr %[[RV]], align 4
// CHECK:   ret i32 %[[R]]
// CHECK: [[LABEL3:.*]]:
// CHECK:   br label %[[LABEL4:.*]]
// CHECK: [[LABEL4]]:
// CHECK:   call void @llvm.trap()
// CHECK:   unreachable
// CHECK: }

long longfunc() { return 42l; }
// CHECK: define{{.*}} i64 @_Z8longfuncv(){{.*}} {
// CHECK:   %[[RV:.*]] = alloca i64, i64 1, align 8
// CHECK:   store i64 42, ptr %[[RV]], align 8
// CHECK:   %[[R:.*]] = load i64, ptr %[[RV]], align 8
// CHECK:   ret i64 %[[R]]
// CHECK: }

unsigned unsignedfunc() { return 42u; }
// CHECK: define{{.*}} i32 @_Z12unsignedfuncv(){{.*}} {
// CHECK:   %[[RV:.*]] = alloca i32, i64 1, align 4
// CHECK:   store i32 42, ptr %[[RV]], align 4
// CHECK:   %[[R:.*]] = load i32, ptr %[[RV]], align 4
// CHECK:   ret i32 %[[R]]
// CHECK: }

unsigned long long ullfunc() { return 42ull; }
// CHECK: define{{.*}} i64 @_Z7ullfuncv(){{.*}} {
// CHECK:   %[[RV:.*]] = alloca i64, i64 1, align 8
// CHECK:   store i64 42, ptr %[[RV]], align 8
// CHECK:   %[[R:.*]] = load i64, ptr %[[RV]], align 8
// CHECK:   ret i64 %[[R]]
// CHECK: }

bool boolfunc() { return true; }
// CHECK: define{{.*}} i1 @_Z8boolfuncv(){{.*}} {
// CHECK:   %[[RV:.*]] = alloca i8, i64 1, align 1
// CHECK:   store i8 1, ptr %[[RV]], align 1
// CHECK:   %[[R8:.*]] = load i8, ptr %[[RV]], align 1
// CHECK:   %[[R:.*]] = trunc i8 %[[R8]] to i1
// CHECK:   ret i1 %[[R]]
// CHECK: }
