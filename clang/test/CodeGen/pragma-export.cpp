// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -x c++ %s -emit-llvm -triple s390x-none-zos -fzos-extensions -fvisibility=hidden -o - | FileCheck %s

// Testing pragma export after decl.
void f0(void) {}
int v0;
#pragma export(f0(void))
#pragma export(v0)

// Testing pragma export before decl.
#pragma export(f1(void))
#pragma export(v1)
void f1(void) {}
int v1;

// Testing overloaded functions.
#pragma export(f2(double, double))
#pragma export(f2(int))
void f2(double, double) {}
void f2(int) {}
void f2(int, int) {}

void f3(double) {}
void f3(int, double) {}
void f3(double, double) {}
#pragma export(f3(double))
#pragma export(f3(int, double))

void f2(void) {}

void t0(void) {
  f2();
}

// Test type decay in arguments

#pragma export(fd1(int[]))
#pragma export(fd2(int*))
#pragma export(fd3(int[]))
#pragma export(fd4(int*))
void fd1(int []) { }
void fd2(int []) { }
void fd3(int *) { }
void fd4(int *) { }


#pragma export (fd5(int ()))
#pragma export (fd6(int (*)()))
#pragma export (fd7(int ()))
#pragma export (fd8(int (*)()))
void fd5(int ()) {}
void fd6(int ()) {}
void fd7(int (*)()) {}
void fd8(int (*)()) {}


// Testing pragma export after decl and usage.
#pragma export(f2(void))

// Testing pragma export with namespace.
void f5(void) {}
void f5a(void) {}
#pragma export(N0::f2a(void))
namespace N0 {
void f0(void) {}
void f1(void) {}
void f2(void) {}
void f3(void) {}
void f5(void) {}
#pragma export(f0(void))
#pragma export(N0::f1(void))
#pragma export(f5(void))
#pragma export(f0a(void))
#pragma export(N0::f1a(void))
#pragma export(f5a(void))
void f0a(void) {}
void f1a(void) {}
void f2a(void) {}
void f3a(void) {}
void f5a(void) {}
} // namespace N0
#pragma export(N0::f2(void))

void f10(int);
#pragma export(f10)
extern "C" void f10(double) {}
void f10(int) {}

// CHECK: @v0 = hidden global i32 0
// CHECK: @v1 = global i32 0
// CHECK: define hidden void @_Z2f0v()
// CHECK: define void @_Z2f1v()
// CHECK: define void @_Z2f2dd(double noundef %0, double noundef %1)
// CHECK: define void @_Z2f2i(i32 noundef signext %0)
// CHECK: define hidden void @_Z2f2ii(i32 noundef signext %0, i32 noundef signext %1)
// CHECK: define hidden void @_Z2f3d(double noundef %0)
// CHECK: define hidden void @_Z2f3id(i32 noundef signext %0, double noundef %1)
// CHECK: define hidden void @_Z2f3dd(double noundef %0, double noundef %1)
// CHECK: define hidden void @_Z2f2v()
// CHECK: define hidden void @_Z2t0v()
// CHECK: define void @_Z3fd1Pi(ptr noundef %0)
// CHECK: define void @_Z3fd2Pi(ptr noundef %0)
// CHECK: define void @_Z3fd3Pi(ptr noundef %0)
// CHECK: define void @_Z3fd4Pi(ptr noundef %0)
// CHECK: define void @_Z3fd5PFivE(ptr noundef %0)
// CHECK: define void @_Z3fd6PFivE(ptr noundef %0)
// CHECK: define void @_Z3fd7PFivE(ptr noundef %0)
// CHECK: define void @_Z3fd8PFivE(ptr noundef %0)
// CHECK: define hidden void @_Z2f5v()
// CHECK: define hidden void @_Z3f5av()
// CHECK: define hidden void @_ZN2N02f0Ev()
// CHECK: define hidden void @_ZN2N02f1Ev()
// CHECK: define hidden void @_ZN2N02f2Ev()
// CHECK: define hidden void @_ZN2N02f3Ev()
// CHECK: define hidden void @_ZN2N02f5Ev()
// CHECK: define void @_ZN2N03f0aEv()
// CHECK: define hidden void @_ZN2N03f1aEv()
// CHECK: define void @_ZN2N03f2aEv()
// CHECK: define hidden void @_ZN2N03f3aEv()
// CHECK: define void @_ZN2N03f5aEv()
// CHECK: define void @f10(double noundef %0) #0 {
// CHECK: define hidden void @_Z3f10i(i32 noundef signext %0) #0 {
