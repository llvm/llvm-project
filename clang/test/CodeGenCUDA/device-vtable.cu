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

// Implicit H+D virtual dtor of an explicit instantiation with a safe body:
// vtable slots reference the real dtor mangled name. No trap.
template <typename T>
struct ETI {
  virtual ~ETI() = default;
};
template class ETI<float>;
// CHECK-DEVICE: @_ZTV3ETIIfE =
// CHECK-DEVICE-SAME: @_ZN3ETIIfED1Ev
// CHECK-DEVICE-SAME: @_ZN3ETIIfED0Ev
// CHECK-HOST: @_ZTV3ETIIfE = {{.*}} @_ZN3ETIIfED

// Device uses ETI_Used (local var): vtable D1 slot holds the real dtor.
template <typename T>
struct ETI_Used {
  virtual ~ETI_Used() = default;
};
template class ETI_Used<float>;
__device__ void use_eti_used() { ETI_Used<float> x; }
// CHECK-DEVICE: @_ZTV8ETI_UsedIfE = {{.*}} @_ZN8ETI_UsedIfED1Ev
// CHECK-HOST: @_ZTV8ETI_UsedIfE = {{.*}} @_ZN8ETI_UsedIfED

// Explicit __device__ dtor: heuristic only gates on implicit H+D, so
// the device vtable gets the real dtor pointers.
template <typename T>
struct ETI_Dev {
  virtual __device__ ~ETI_Dev() = default;
};
template class ETI_Dev<float>;
// CHECK-DEVICE: @_ZTV7ETI_DevIfE = {{.*}} @_ZN7ETI_DevIfED

// Heap-allocate and polymorphically delete on device: the heuristic
// must detect device use via the ctor and emit real dtor pointers so
// `delete pBase` dispatches to a real body, not a trap stub.
__device__ void *operator new(__SIZE_TYPE__);
__device__ void operator delete(void *);
struct ETI_Base {
  virtual ~ETI_Base() = default;
};
template <typename T>
struct ETI_Poly : ETI_Base {
  virtual ~ETI_Poly() = default;
  T x;
};
template class ETI_Poly<float>;
__device__ ETI_Base *eti_make() { return new ETI_Poly<float>; }
__device__ void eti_destroy(ETI_Base *p) { delete p; }
// CHECK-DEVICE: @_ZTV8ETI_PolyIfE =
// CHECK-DEVICE-SAME: @_ZN8ETI_PolyIfED1Ev
// CHECK-DEVICE-SAME: @_ZN8ETI_PolyIfED0Ev
// CHECK-HOST: @_ZTV8ETI_PolyIfE = {{.*}} @_ZN8ETI_PolyIfED

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

// No more separately-named trap stub — vtable slots reference the real
// dtor symbol directly. If the deferred-diag walker found the dtor's
// body unsafe, CodeGen replaces that body itself with trap (covered by
// a separate test).
// CHECK-DEVICE-NOT: __clang_cuda_unreachable_dtor

// ETI_Poly D1/D0 bodies must be emitted on device.
// CHECK-DEVICE: define{{.*}} @_ZN8ETI_PolyIfED1Ev
// CHECK-DEVICE: define{{.*}} @_ZN8ETI_PolyIfED0Ev

