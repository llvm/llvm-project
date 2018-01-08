// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
int __attribute__((target("sse4.2"))) foo(int i) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(int);
int __attribute__((target("arch=ivybridge"))) foo(int i) {return 1;}
int __attribute__((target("default"))) foo(int i) { return 2; }

typedef int (*FuncPtr)(int);
void func(FuncPtr);

int bar() {
  func(foo);
  FuncPtr Free = &foo;
  FuncPtr Free2 = foo;

  return 0;
  return Free(1) + Free(2);
}

// CHECK: @foo.ifunc = ifunc i32 (i32), i32 (i32)* ()* @foo.resolver
// CHECK: define i32 @foo.sse4.2(
// CHECK: ret i32 0
// CHECK: define i32 @foo.arch_ivybridge(
// CHECK: ret i32 1
// CHECK: define i32 @foo(
// CHECK: ret i32 2

// CHECK: define i32 @bar()
// CHECK: call void @func(i32 (i32)* @foo.ifunc)
// CHECK: store i32 (i32)* @foo.ifunc
// CHECK: store i32 (i32)* @foo.ifunc

// CHECK: declare i32 @foo.arch_sandybridge(
