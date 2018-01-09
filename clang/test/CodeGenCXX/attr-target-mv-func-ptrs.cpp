// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
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


// CHECK: @_Z3fooi.ifunc 
// CHECK: @_ZN1S3fooEi.ifunc

// CHECK: define i32 @_Z3barv()
// Store to Free of ifunc
// CHECK: store i32 (i32)* @_Z3fooi.ifunc
// Store to Member of ifunc
// CHECK: store { i64, i64 } { i64 ptrtoint (i32 (%struct.S*, i32)* @_ZN1S3fooEi.ifunc to i64), i64 0 }, { i64, i64 }* [[MEMBER:%[a-z]+]]

// Call to 'f' with the ifunc
// CHECK: call void @_Z1fPFiiEM1SFiiE(i32 (i32)* @_Z3fooi.ifunc
