// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// Make sure we don't emit vtables for classes with methods that have
// inappropriate target attributes. Currently it's mostly needed in
// order to avoid emitting vtables for host-only classes on device
// side where we can't codegen them.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:     | FileCheck %s -check-prefix=CHECK-HOST -check-prefix=CHECK-BOTH
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm -o - %s \
// RUN:     | FileCheck %s -check-prefix=CHECK-DEVICE -check-prefix=CHECK-BOTH

#include "Inputs/cuda.h"

struct H  {
  virtual void method();
};
//CHECK-HOST: @_ZTV1H =
//CHECK-HOST-SAME: @_ZN1H6methodEv
//CHECK-DEVICE-NOT: @_ZTV1H =
//CHECK-DEVICE-NOT: @_ZTVN10__cxxabiv117__class_type_infoE
//CHECK-DEVICE-NOT: @_ZTS1H
//CHECK-DEVICE-NOT: @_ZTI1H
struct D  {
   __device__ virtual void method();
};

//CHECK-DEVICE: @_ZTV1D
//CHECK-DEVICE-SAME: @_ZN1D6methodEv
//CHECK-HOST-NOT: @_ZTV1D
//CHECK-DEVICE-NOT: @_ZTVN10__cxxabiv117__class_type_infoE
//CHECK-DEVICE-NOT: @_ZTS1D
//CHECK-DEVICE-NOT: @_ZTI1D
// This is the case with mixed host and device virtual methods.  It's
// impossible to emit a valid vtable in that case because only host or
// only device methods would be available during host or device
// compilation. At the moment Clang (and NVCC) emit NULL pointers for
// unavailable methods,
struct HD  {
  virtual void h_method();
  __device__ virtual void d_method();
};
// CHECK-BOTH: @_ZTV2HD
// CHECK-DEVICE-NOT: @_ZN2HD8h_methodEv
// CHECK-DEVICE-SAME: null
// CHECK-DEVICE-SAME: @_ZN2HD8d_methodEv
// CHECK-HOST-SAME: @_ZN2HD8h_methodEv
// CHECK-HOST-NOT: @_ZN2HD8d_methodEv
// CHECK-HOST-SAME: null
// CHECK-BOTH-SAME: ]
// CHECK-DEVICE-NOT: @_ZTVN10__cxxabiv117__class_type_infoE
// CHECK-DEVICE-NOT: @_ZTS2HD
// CHECK-DEVICE-NOT: @_ZTI2HD

void H::method() {}
//CHECK-HOST: define void @_ZN1H6methodEv

void __device__ D::method() {}
//CHECK-DEVICE: define void @_ZN1D6methodEv

void __device__ HD::d_method() {}
// CHECK-DEVICE: define void @_ZN2HD8d_methodEv
// CHECK-HOST-NOT: define void @_ZN2HD8d_methodEv
void HD::h_method() {}
// CHECK-HOST: define void @_ZN2HD8h_methodEv
// CHECK-DEVICE-NOT: define void @_ZN2HD8h_methodEv

