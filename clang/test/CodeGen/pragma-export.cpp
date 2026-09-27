// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -x c++ %s -emit-llvm -triple s390x-none-zos -fzos-extensions -fvisibility=hidden -o - | FileCheck %s

// Testing pragma export after decl.
extern "C" void f0(void) {}
int v0;
#pragma export(f0)
#pragma export(v0)

// Testing pragma export before decl.
#pragma export(f1)
#pragma export(v1)
extern "C" void f1(void) {}
int v1;

// Testing overloaded functions.
#pragma export(f2)
void f2(double, double) {}
extern "C" void f2(int) {}
void f2(int, int) {}

extern "C" void f3(double) {}
void f3(int, double) {}
void f3(double, double) {}
#pragma export(f3)

extern "C" void f2b(void) {}

void t0(void) {
  f2b();
}

// Testing pragma export after decl and usage.
#pragma export(f2b)

// Testing pragma export with namespace.
extern "C" void f5(void) {}
extern "C" void f5a(void) {}
namespace N0 {
void f5(void) {}
#pragma export(f5)
#pragma export(f5a)
void f5a(void) {}
} // namespace N0

void f10(int);
#pragma export(f10)
extern "C" void f10(double) {}
void f10(int) {}

// CHECK: @v0 = hidden global i32 0
// CHECK: @v1 = global i32 0
// CHECK: define hidden void @f0()
// CHECK: define void @f1()
// CHECK: define hidden void @_Z2f2dd(double noundef %0, double noundef %1)
// CHECK: define void @f2(i32 noundef signext %0)
// CHECK: define hidden void @_Z2f2ii(i32 noundef signext %0, i32 noundef signext %1)
// CHECK: define hidden void @f3(double noundef %0)
// CHECK: define hidden void @_Z2f3id(i32 noundef signext %0, double noundef %1)
// CHECK: define hidden void @_Z2f3dd(double noundef %0, double noundef %1)
// CHECK: define hidden void @f2b()
// CHECK: define hidden void @_Z2t0v()
// CHECK: define hidden void @f5()
// CHECK: define hidden void @f5a()
// CHECK: define hidden void @_ZN2N02f5Ev()
// CHECK: define hidden void @_ZN2N03f5aEv()
// CHECK: define void @f10(double noundef %0)
// CHECK: define hidden void @_Z3f10i(i32 noundef signext %0)

