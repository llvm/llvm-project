// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -O0 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DEVICE,PTR64
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -O2 -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DEVICE,PTR64
// RUN: %clang_cc1 -triple nvptx-nvidia-cuda -fcuda-is-device -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DEVICE,PTR32
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=HOST

#include "Inputs/cuda.h"

extern "C" __device__ void escape(const void *);

struct S {
  int x, y;
};

// Functions of the same source type must not share ABI classifications when
// their parameters have different grid-constant attributes.
// DEVICE-LABEL: define{{.*}} ptx_kernel void @plain_before(i32 noundef %arg)
// DEVICE: %arg.addr = alloca i32, align 4
// DEVICE: store i32 %arg, ptr %arg.addr, align 4
// DEVICE: call void @escape(ptr noundef %arg.addr)
extern "C" __global__ void plain_before(const int arg) {
  escape(&arg);
}

// DEVICE-LABEL: define{{.*}} ptx_kernel void @grid_after(ptr noundef byval(i32) align 4 "nvvm.grid_constant" %arg)
// DEVICE-NOT: alloca
// DEVICE: call void @escape(ptr noundef %arg)
// DEVICE-NEXT: ret void
// HOST-LABEL: define{{.*}} void @__device_stub__grid_after(i32 noundef %arg)
extern "C" __global__ void grid_after(const __grid_constant__ int arg) {
  escape(&arg);
}

// DEVICE-LABEL: define{{.*}} ptx_kernel void @plain_after(i32 noundef %arg)
// DEVICE: %arg.addr = alloca i32, align 4
// DEVICE: call void @escape(ptr noundef %arg.addr)
extern "C" __global__ void plain_after(const int arg) {
  escape(&arg);
}

// All three addresses must refer to the incoming byval parameter objects.
// DEVICE-LABEL: define{{.*}} ptx_kernel void @mixed(ptr noundef byval(%struct.S) align 4 "nvvm.grid_constant" %s, ptr noundef byval(i32) align 4 "nvvm.grid_constant" %scalar, ptr noundef byval(%struct.char4) align 1 "nvvm.grid_constant" %vec)
// DEVICE-NOT: alloca
// DEVICE: call void @escape(ptr noundef %s)
// DEVICE-NEXT: call void @escape(ptr noundef %scalar)
// DEVICE-NEXT: call void @escape(ptr noundef %vec)
// DEVICE-NEXT: ret void
extern "C" __global__ void mixed(const __grid_constant__ S s,
                                 const __grid_constant__ int scalar,
                                 const __grid_constant__ char4 vec) {
  escape(&s);
  escape(&scalar);
  escape(&vec);
}

// The grid-constant argument positions are part of the ABI cache key.
// DEVICE-LABEL: define{{.*}} ptx_kernel void @grid_first(ptr noundef byval(i32) align 4 "nvvm.grid_constant" %a, i32 noundef %b)
// DEVICE: %b.addr = alloca i32, align 4
// DEVICE: call void @escape(ptr noundef %a)
// DEVICE-NEXT: call void @escape(ptr noundef %b.addr)
extern "C" __global__ void grid_first(const __grid_constant__ int a, const int b) {
  escape(&a);
  escape(&b);
}

// DEVICE-LABEL: define{{.*}} ptx_kernel void @grid_second(i32 noundef %a, ptr noundef byval(i32) align 4 "nvvm.grid_constant" %b)
// DEVICE: %a.addr = alloca i32, align 4
// DEVICE: call void @escape(ptr noundef %a.addr)
// DEVICE-NEXT: call void @escape(ptr noundef %b)
extern "C" __global__ void grid_second(const int a, const __grid_constant__ int b) {
  escape(&a);
  escape(&b);
}

enum E : unsigned short { E0 };
using V = int __attribute__((ext_vector_type(4)));

// DEVICE-LABEL: define{{.*}} ptx_kernel void @scalar_types(
// DEVICE-SAME: ptr noundef byval(i8) align 1 "nvvm.grid_constant" %c,
// DEVICE-SAME: ptr noundef byval(i16) align 2 "nvvm.grid_constant" %h,
// DEVICE-SAME: ptr noundef byval(i8) align 1 "nvvm.grid_constant" %b,
// DEVICE-SAME: ptr noundef byval(float) align 4 "nvvm.grid_constant" %f,
// DEVICE-SAME: ptr noundef byval(double) align 8 "nvvm.grid_constant" %d,
// PTR64-SAME: ptr noundef byval(ptr) align 8 "nvvm.grid_constant" %p,
// PTR32-SAME: ptr noundef byval(ptr) align 4 "nvvm.grid_constant" %p,
// DEVICE-SAME: ptr noundef byval(i16) align 2 "nvvm.grid_constant" %e,
// DEVICE-SAME: ptr noundef byval(<4 x i32>) align 16 "nvvm.grid_constant" %v)
// DEVICE-NOT: alloca
// DEVICE: call void @escape(ptr noundef %c)
// DEVICE-NEXT: call void @escape(ptr noundef %h)
// DEVICE-NEXT: call void @escape(ptr noundef %b)
// DEVICE-NEXT: call void @escape(ptr noundef %f)
// DEVICE-NEXT: call void @escape(ptr noundef %d)
// DEVICE-NEXT: call void @escape(ptr noundef %p)
// DEVICE-NEXT: call void @escape(ptr noundef %e)
// DEVICE-NEXT: call void @escape(ptr noundef %v)
// DEVICE-NEXT: ret void
extern "C" __global__ void scalar_types(
    const __grid_constant__ char c, const __grid_constant__ short h,
    const __grid_constant__ bool b, const __grid_constant__ float f,
    const __grid_constant__ double d, int *const __grid_constant__ p,
    const __grid_constant__ E e, const __grid_constant__ V v) {
  escape(&c);
  escape(&h);
  escape(&b);
  escape(&f);
  escape(&d);
  escape(&p);
  escape(&e);
  escape(&v);
}

// Values are loaded from the parameter objects, while a source-level copy is
// still a distinct local object.
// DEVICE-LABEL: define{{.*}} ptx_kernel void @values(
// DEVICE: %copy = alloca i32, align 4
// DEVICE: load i32, ptr %s, align 4
// DEVICE: store i32 {{.*}}, ptr %copy, align 4
// DEVICE: call void @escape(ptr noundef %s)
// DEVICE-NEXT: call void @escape(ptr noundef %copy)
// DEVICE: load i8, ptr %b, align 1
// DEVICE: load <4 x i32>, ptr %v, align 16
// DEVICE: load ptr, ptr %p, align
extern "C" __global__ void values(
    const __grid_constant__ int s, const __grid_constant__ bool b,
    int *const __grid_constant__ p, const __grid_constant__ V v) {
  const int copy = s;
  escape(&s);
  escape(&copy);
  if (b)
    *p = v.x;
}

// A prior declaration and template instantiation must use the same byval ABI.
extern "C" __global__ void declared(const __grid_constant__ int arg);
void launch_declared() {
  declared<<<1, 1>>>(1);
}
// DEVICE-LABEL: define{{.*}} ptx_kernel void @declared(ptr noundef byval(i32) align 4 "nvvm.grid_constant" %arg)
// DEVICE-NOT: alloca
// DEVICE: call void @escape(ptr noundef %arg)
extern "C" __global__ void declared(const __grid_constant__ int arg) {
  escape(&arg);
}

// DEVICE-LABEL: define{{.*}} ptx_kernel void @_ZN7Kernels6memberEi(ptr noundef byval(i32) align 4 "nvvm.grid_constant" %arg)
// DEVICE-NOT: alloca
// DEVICE: call void @escape(ptr noundef %arg)
struct Kernels {
  static __global__ void member(const __grid_constant__ int arg) {
    escape(&arg);
  }
};

// DEVICE-LABEL: define{{.*}} ptx_kernel void @_Z15template_kernelIiEvT_(ptr noundef byval(i32) align 4 "nvvm.grid_constant" %arg)
// DEVICE-NOT: alloca
// DEVICE: call void @escape(ptr noundef %arg)
template <typename T>
__global__ void template_kernel(const __grid_constant__ T arg) {
  escape(&arg);
}

void launch() {
  Kernels::member<<<1, 1>>>(1);
  template_kernel<int><<<1, 1>>>(1);
}
