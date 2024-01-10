// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-macos -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS
void temp();
void temp(int);
using FP = void(*)(int);
void b() {
  FP f = temp; 
}

int __attribute__((target("sse4.2"))) foo(int) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(int);
int __attribute__((target("arch=ivybridge"))) foo(int) {return 1;}
int __attribute__((target("default"))) foo(int) { return 2; }

struct S {
int __attribute__((target("sse4.2"))) foo(int) { return 0; }
int __attribute__((target("arch=sandybridge"))) foo(int);
int __attribute__((target("arch=ivybridge"))) foo(int) {return 1;}
int __attribute__((target("default"))) foo(int) { return 2; }
};

using FuncPtr = int (*)(int);
using MemFuncPtr = int (S::*)(int);

void f(FuncPtr, MemFuncPtr);

int bar() {
  FuncPtr Free = &foo;
  MemFuncPtr Member = &S::foo;
  S s;
  f(foo, &S::foo);
  return Free(1) + (s.*Member)(2);
}

// LINUX: @_Z3fooi.ifunc
// LINUX: @_ZN1S3fooEi.ifunc

// LINUX: define{{.*}} i32 @_Z3barv()
// Store to Free of ifunc
// LINUX: store ptr @_Z3fooi.ifunc
// Store to Member of ifunc
// LINUX: store { i64, i64 } { i64 ptrtoint (ptr @_ZN1S3fooEi.ifunc to i64), i64 0 }, ptr [[MEMBER:%[a-z]+]]

// Call to 'f' with the ifunc
// LINUX: call void @_Z1fPFiiEM1SFiiE(ptr noundef @_Z3fooi.ifunc

// WINDOWS: define dso_local noundef i32 @"?bar@@YAHXZ"()
// Store to Free
// WINDOWS: store ptr @"?foo@@YAHH@Z.resolver", ptr
// Store to Member
// WINDOWS: store ptr @"?foo@S@@QEAAHH@Z.resolver", ptr

// Call to 'f'
// WINDOWS: call void @"?f@@YAXP6AHH@ZP8S@@EAAHH@Z@Z"(ptr noundef @"?foo@@YAHH@Z.resolver", ptr @"?foo@S@@QEAAHH@Z.resolver")
