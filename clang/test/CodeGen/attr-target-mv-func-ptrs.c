// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS
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

// LINUX: @foo.ifunc = ifunc i32 (i32), i32 (i32)* ()* @foo.resolver
// LINUX: define i32 @foo.sse4.2(
// LINUX: ret i32 0
// LINUX: define i32 @foo.arch_ivybridge(
// LINUX: ret i32 1
// LINUX: define i32 @foo(
// LINUX: ret i32 2

// WINDOWS: define dso_local i32 @foo.sse4.2(
// WINDOWS: ret i32 0
// WINDOWS: define dso_local i32 @foo.arch_ivybridge(
// WINDOWS: ret i32 1
// WINDOWS: define dso_local i32 @foo(
// WINDOWS: ret i32 2

// LINUX: define i32 @bar()
// LINUX: call void @func(i32 (i32)* @foo.ifunc)
// LINUX: store i32 (i32)* @foo.ifunc
// LINUX: store i32 (i32)* @foo.ifunc

// WINDOWS: define dso_local i32 @bar()
// WINDOWS: call void @func(i32 (i32)* @foo.resolver)
// WINDOWS: store i32 (i32)* @foo.resolver
// WINDOWS: store i32 (i32)* @foo.resolver

// LINUX: declare i32 @foo.arch_sandybridge(

// WINDOWS: declare dso_local i32 @foo.arch_sandybridge(
