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
// CHECK:   ret i32 42

int scopes() {
  {
    {
      return 99;
    }
  }
}
// CHECK: define{{.*}} i32 @_Z6scopesv(){{.*}} {
// CHECK:   br label %[[LABEL1:.*]]
// CHECK: [[LABEL1]]:
// CHECK:   ret i32 99
// CHECK: [[LABEL3:.*]]:
// CHECK:   call void @llvm.trap()
// CHECK:   unreachable
// CHECK: }

long longfunc() { return 42l; }
// CHECK: define{{.*}} i64 @_Z8longfuncv(){{.*}} {
// CHECK:   ret i64 42
// CHECK: }

unsigned unsignedfunc() { return 42u; }
// CHECK: define{{.*}} i32 @_Z12unsignedfuncv(){{.*}} {
// CHECK:   ret i32 42
// CHECK: }

unsigned long long ullfunc() { return 42ull; }
// CHECK: define{{.*}} i64 @_Z7ullfuncv(){{.*}} {
// CHECK:   ret i64 42
// CHECK: }

bool boolfunc() { return true; }
// CHECK: define{{.*}} i1 @_Z8boolfuncv(){{.*}} {
// CHECK:   ret i1 true
// CHECK: }
