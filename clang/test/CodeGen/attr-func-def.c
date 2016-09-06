// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10.0 -emit-llvm -Oz -o - %s | FileCheck %s

// CHECK: define i32 @foo2(i32 %a) local_unnamed_addr [[ATTRS2:#[0-9]+]] {
// CHECK: define i32 @foo1(i32 %a) local_unnamed_addr [[ATTRS1:#[0-9]+]] {

int foo1(int);

int foo2(int a) {
  return foo1(a + 2);
}

__attribute__((optnone))
int foo1(int a) {
    return a + 1;
}

// CHECK: attributes [[ATTRS2]] = { {{.*}}optsize{{.*}} }
// CHECK: attributes [[ATTRS1]] = { {{.*}}optnone{{.*}} }
