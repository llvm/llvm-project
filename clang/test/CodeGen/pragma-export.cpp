// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -x c++ %s -emit-llvm -triple s390x-none-zos -fzos-extensions -fvisibility=hidden -o - | FileCheck %s

// Testing pragma export after decl.
void f0(void) {}
int v0;
#pragma export(f0)
#pragma export(v0)

// Testing pragma export before decl.
#pragma export(f1)
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

void fd1(int []) { }
void fd2(int []) { }
void fd3(int *) { }
void fd4(int *) { }

#pragma export(fd1(int[]))
#pragma export(fd2(int*))
#pragma export(fd3(int[]))
#pragma export(fd4(int*))

void fd5(int ()) {}
void fd6(int ()) {}
void fd7(int (*)()) {}
void fd8(int (*)()) {}

#pragma export (fd5(int ()))
#pragma export (fd6(int (*)()))
#pragma export (fd7(int ()))
#pragma export (fd8(int (*)()))

// Testing pragma export after decl and usage.
#pragma export(f2(void))

// Testing pragma export with namespace.
void f5(void) {}
namespace N0 {
void f0(void) {}
void f1(void) {}
void f2(void) {}
void f3(void) {}
void f5(void) {}
#pragma export(f0)
#pragma export(N0::f1)
#pragma export(f5)
} // namespace N0
#pragma export(N0::f2)

// CHECK: @v0 = global i32
// CHECK: @v1 = global i32
// CHECK: define void @_Z2f0v
// CHECK: define void @_Z2f1v
// CHECK: define void @_Z2f2dd
// CHECK: define void @_Z2f2i
// CHECK: define hidden void @_Z2f2ii
// CHECK: define void @_Z2f3d
// CHECK: define void @_Z2f3id
// CHECK: define hidden void @_Z2f3dd
// CHECK: define void @_Z2f2v
// CHECK: define hidden void @_Z2t0v
// CHECK: define void @_Z3fd1Pi
// CHECK: define void @_Z3fd2Pi
// CHECK: define void @_Z3fd3Pi
// CHECK: define void @_Z3fd4Pi
// CHECK: define void @_Z3fd5PFivE
// CHECK: define void @_Z3fd6PFivE
// CHECK: define void @_Z3fd7PFivE
// CHECK: define void @_Z3fd8PFivE
// CHECK: define hidden void @_Z2f5v
// CHECK: define void @_ZN2N02f0Ev
// CHECK: define void @_ZN2N02f1Ev
// CHECK: define void @_ZN2N02f2Ev
// CHECK: define hidden void @_ZN2N02f3Ev
// CHECK: define void @_ZN2N02f5Ev
