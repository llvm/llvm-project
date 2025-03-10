// Simple functions
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o -  | FileCheck %s

void empty() { }
// CHECK: define{{.*}} void @empty()
// CHECK:   ret void

void voidret() { return; }
// CHECK: define{{.*}} void @voidret()
// CHECK:   ret void

int intfunc() { return 42; }
// CHECK: define{{.*}} i32 @intfunc()
// CHECK:   ret i32 42

int scopes() {
  {
    {
      return 99;
    }
  }
}
// CHECK: define{{.*}} i32 @scopes() {
// CHECK:     br label %[[LABEL1:.*]]
// CHECK:     [[LABEL1]]:
// CHECK:       br label %[[LABEL2:.*]]
// CHECK:     [[LABEL2]]:
// CHECK:       ret i32 99
// CHECK:     [[LABEL3:.*]]:
// CHECK:       br label %[[LABEL4:.*]]
// CHECK:     [[LABEL4]]:
// CHECK:       call void @llvm.trap()
// CHECK:       unreachable
// CHECK: }

long longfunc() { return 42l; }
// CHECK: define{{.*}} i64 @longfunc() {
// CHECK:   ret i64 42
// CHECK: }

unsigned unsignedfunc() { return 42u; }
// CHECK: define{{.*}} i32 @unsignedfunc() {
// CHECK:   ret i32 42
// CHECK: }

unsigned long long ullfunc() { return 42ull; }
// CHECK: define{{.*}} i64 @ullfunc() {
// CHECK:   ret i64 42
// CHECK: }

bool boolfunc() { return true; }
// CHECK: define{{.*}} i1 @boolfunc() {
// CHECK:   ret i1 true
// CHECK: }
