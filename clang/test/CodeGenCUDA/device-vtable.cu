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

// Explicit class template instantiation with an implicit __host__ __device__
// virtual destructor: when no device code references the class, the device
// vtable should fill destructor slots with NULL so that emitting the
// destructor body (which may reach host-only callees through e.g.
// libstdc++ runtime dispatch) is not forced. This block is checked first
// because comdat globals are emitted before non-comdat globals in IR.
template <typename T>
struct ETI {
  virtual ~ETI() = default;
};
template class ETI<float>;
// CHECK-DEVICE: @_ZTV3ETIIfE = {{.*}} zeroinitializer
// CHECK-HOST: @_ZTV3ETIIfE = {{.*}} @_ZN3ETIIfED

// Device code does reference ETI_Used<float>: the per-slot NULL extension
// must NOT fire — the device vtable's complete-destructor slot must hold
// the real pointer (the deleting-destructor slot stays unused because no
// device code performs `delete`).
template <typename T>
struct ETI_Used {
  virtual ~ETI_Used() = default;
};
template class ETI_Used<float>;
__device__ void use_eti_used() { ETI_Used<float> x; }
// CHECK-DEVICE: @_ZTV8ETI_UsedIfE = {{.*}} @_ZN8ETI_UsedIfED1Ev
// CHECK-HOST: @_ZTV8ETI_UsedIfE = {{.*}} @_ZN8ETI_UsedIfED

// Explicit __device__ virtual destructor on an explicit instantiation:
// the per-slot NULL extension must NOT fire (it gates on implicit H+D),
// so the device vtable holds the real destructor pointers.
template <typename T>
struct ETI_Dev {
  virtual __device__ ~ETI_Dev() = default;
};
template class ETI_Dev<float>;
// CHECK-DEVICE: @_ZTV7ETI_DevIfE = {{.*}} @_ZN7ETI_DevIfED

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
//CHECK-HOST: define{{.*}} void @_ZN1H6methodEv

void __device__ D::method() {}
//CHECK-DEVICE: define{{.*}} void @_ZN1D6methodEv

void __device__ HD::d_method() {}
// CHECK-DEVICE: define{{.*}} void @_ZN2HD8d_methodEv
// CHECK-HOST-NOT: define{{.*}} void @_ZN2HD8d_methodEv
void HD::h_method() {}
// CHECK-HOST: define{{.*}} void @_ZN2HD8h_methodEv
// CHECK-DEVICE-NOT: define{{.*}} void @_ZN2HD8h_methodEv

