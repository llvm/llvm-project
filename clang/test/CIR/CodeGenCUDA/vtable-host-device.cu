#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s --check-prefix=CIR-DEV --implicit-check-not=@_ZNK6Square4nameEv --implicit-check-not=@_ZN1C9host_onlyEv
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=DEV --implicit-check-not=@_ZNK6Square4nameEv --implicit-check-not=@_ZN1C9host_onlyEv
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=DEV --implicit-check-not=@_ZNK6Square4nameEv --implicit-check-not=@_ZN1C9host_onlyEv
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=DEV --implicit-check-not=@_ZNK6Square4nameEv --implicit-check-not=@_ZN1C9host_onlyEv
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=DEV --implicit-check-not=@_ZNK6Square4nameEv --implicit-check-not=@_ZN1C9host_onlyEv

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=HOST --implicit-check-not=@_ZNK6Square4areaEv
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefix=HOST --implicit-check-not=@_ZNK6Square4areaEv

// A vtable has null in the slots of virtual functions that can't be compiled
// on the current side, so those functions aren't referenced or emitted there.
// On the device, pure and deleted virtual functions are weak functions that
// trap, since no device library provides them.

struct Shape {
  __host__ __device__ virtual ~Shape() {}
  __device__ virtual float area() const = 0;
  virtual const char *name() const = 0;
  __host__ __device__ virtual void removed() = delete;
};

struct Square : Shape {
  float s;
  __host__ __device__ Square(float s) : s(s) {}
  __device__ float area() const override { return s * s; }
  const char *name() const override { return "square"; }
};

__global__ void kernel(float *out) {
  Square sq(3.0f);
  *out = sq.area();
}

const char *host_name() { return Square(3.0f).name(); }

// CIR-DEV: cir.global {{.*}} @_ZTV6Square = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZN6SquareD1Ev> : !cir.ptr<!u8i>, #cir.global_view<@_ZN6SquareD0Ev> : !cir.ptr<!u8i>, #cir.global_view<@_ZNK6Square4areaEv> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@__cxa_deleted_virtual> : !cir.ptr<!u8i>]>
// CIR-DEV: cir.global {{.*}} @_ZTV5Shape = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZN5ShapeD1Ev> : !cir.ptr<!u8i>, #cir.global_view<@_ZN5ShapeD0Ev> : !cir.ptr<!u8i>, #cir.global_view<@__cxa_pure_virtual> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@__cxa_deleted_virtual> : !cir.ptr<!u8i>]>

// DEV-DAG: @_ZTV6Square = {{.*}}[ptr {{[^,]*}}null, ptr {{[^,]*}}null, ptr {{[^,]*}}@_ZN6SquareD1Ev{{[^,]*}}, ptr {{[^,]*}}@_ZN6SquareD0Ev{{[^,]*}}, ptr {{[^,]*}}@_ZNK6Square4areaEv{{[^,]*}}, ptr {{[^,]*}}null, ptr {{[^,]*}}@__cxa_deleted_virtual{{[^,]*}}]
// DEV-DAG: @_ZTV5Shape = {{.*}}[ptr {{[^,]*}}null, ptr {{[^,]*}}null, ptr {{[^,]*}}@_ZN5ShapeD1Ev{{[^,]*}}, ptr {{[^,]*}}@_ZN5ShapeD0Ev{{[^,]*}}, ptr {{[^,]*}}@__cxa_pure_virtual{{[^,]*}}, ptr {{[^,]*}}null, ptr {{[^,]*}}@__cxa_deleted_virtual{{[^,]*}}]

// HOST-DAG: @_ZTV6Square = {{.*}}[ptr null, ptr @_ZTI6Square, ptr @_ZN6SquareD1Ev, ptr @_ZN6SquareD0Ev, ptr null, ptr @_ZNK6Square4nameEv, ptr @__cxa_deleted_virtual]
// HOST-DAG: @_ZTV5Shape = {{.*}}[ptr null, ptr @_ZTI5Shape, ptr @_ZN5ShapeD1Ev, ptr @_ZN5ShapeD0Ev, ptr null, ptr @__cxa_pure_virtual, ptr @__cxa_deleted_virtual]
// HOST-DAG: declare {{.*}}void @__cxa_pure_virtual()
// HOST-DAG: declare {{.*}}void @__cxa_deleted_virtual()

// CIR-DEV:      cir.func weak {{.*}}@__cxa_deleted_virtual() {
// CIR-DEV-NEXT:   cir.trap
// CIR-DEV-NEXT: }
// CIR-DEV:      cir.func weak {{.*}}@__cxa_pure_virtual() {
// CIR-DEV-NEXT:   cir.trap
// CIR-DEV-NEXT: }

// DEV-DAG: define weak {{.*}}void @__cxa_pure_virtual()
// DEV-DAG: define weak {{.*}}void @__cxa_deleted_virtual()

// A host-only override that needs a thunk since its slot is null and the thunk
// index still advances, so the next thunk lands in its own slot.
struct A {
  virtual void host_only() = 0;
  __device__ virtual void f1() = 0;
};
struct B {
  __device__ virtual void f2() {}
};
struct C : B, A {
  void host_only() override {}
  __device__ void f1() override {}
};
__device__ void use_c() { C c; }

// DEV-DAG: @_ZTV1C = {{.*}}[ptr {{[^,]*}}null, ptr {{[^,]*}}null, ptr {{[^,]*}}@_ZN1B2f2Ev{{[^,]*}}, ptr {{[^,]*}}null, ptr {{[^,]*}}@_ZN1C2f1Ev{{[^,]*}}], [4 x ptr{{[^]]*}}] [ptr {{[^,]*}}inttoptr (i64 -8 to ptr{{[^,]*}}), ptr {{[^,]*}}null, ptr {{[^,]*}}null, ptr {{[^,]*}}@_ZThn8_N1C2f1Ev{{[^,]*}}]
