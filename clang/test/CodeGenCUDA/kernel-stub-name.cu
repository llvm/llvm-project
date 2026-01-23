// RUN: echo "GPU binary would be here" > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -check-prefixes=CHECK,GNU,GNU-HIP,HIP %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -check-prefix=NEG %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm %s \
// RUN:     -aux-triple amdgcn-amd-amdhsa -fcuda-include-gpubinary \
// RUN:     %t -o - -x hip\
// RUN:   | FileCheck -check-prefixes=CHECK,MSVC,MSVC-HIP,HIP %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm %s \
// RUN:     -aux-triple nvptx64 -fcuda-include-gpubinary \
// RUN:     %t -target-sdk-version=9.2 -o - \
// RUN:   | FileCheck -check-prefixes=CHECK,MSVC,CUDA %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm %s \
// RUN:     -aux-triple amdgcn-amd-amdhsa -fcuda-include-gpubinary \
// RUN:     %t -o - -x hip\
// RUN:   | FileCheck -check-prefix=NEG %s

#include "Inputs/cuda.h"

// Check kernel handles are emitted for non-MSVC target but not for MSVC target.

// GNU-HIP: @[[HCKERN:ckernel]] = constant ptr @[[CSTUB:__device_stub__ckernel]], align 8
// GNU-HIP: @[[HNSKERN:_ZN2ns8nskernelEv]] = constant ptr @[[NSSTUB:_ZN2ns23__device_stub__nskernelEv]], align 8
// GNU-HIP: @[[HTKERN:_Z10kernelfuncIiEvv]] = linkonce_odr constant ptr @[[TSTUB:_Z25__device_stub__kernelfuncIiEvv]], comdat, align 8
// GNU-HIP: @[[HDKERN:_Z11kernel_declv]] = external constant ptr, align 8
// GNU-HIP: @[[HTDKERN:_Z20template_kernel_declIiEvT_]] = external constant ptr, align 8

// MSVC-HIP: @[[HCKERN:ckernel]] = dso_local constant ptr @[[CSTUB:__device_stub__ckernel]], align 8
// MSVC-HIP: @[[HNSKERN:"\?nskernel@ns@@YAXXZ.*"]] = dso_local constant ptr @[[NSSTUB:"\?__device_stub__nskernel@ns@@YAXXZ"]], align 8
// MSVC-HIP: @[[HTKERN:"\?\?\$kernelfunc@H@@YAXXZ.*"]] = linkonce_odr dso_local constant ptr @[[TSTUB:"\?\?\$__device_stub__kernelfunc@H@@YAXXZ.*"]], comdat, align 8
// MSVC-HIP: @[[HDKERN:"\?kernel_decl@@YAXXZ.*"]] = external dso_local constant ptr, align 8
// MSVC-HIP: @[[HTDKERN:"\?\?\$template_kernel_decl@H@@YAXH.*"]] = external dso_local constant ptr, align 8
extern "C" __global__ void ckernel() {}

// CUDA: @[[HCKERN:__device_stub__ckernel\.id]] = dso_local global i8 0
// CUDA: @[[HNSKERN:"\?__device_stub__nskernel@ns@@YAXXZ\.id"]] = dso_local global i8 0
// CUDA: @[[HTKERN:"\?\?\$__device_stub__kernelfunc@H@@YAXXZ\.id"]] = linkonce_odr dso_local global i8 0, comdat

namespace ns {
__global__ void nskernel() {}
} // namespace ns

template<class T>
__global__ void kernelfunc() {}

__global__ void kernel_decl();

template<class T>
__global__ void template_kernel_decl(T x);

extern "C" void (*kernel_ptr)();
extern "C" void *void_ptr;

extern "C" void launch(void *kern);

// Device side kernel names

// CHECK: @[[CKERN:[0-9]*]] = {{.*}} c"ckernel\00"
// CHECK: @[[NSKERN:[0-9]*]] = {{.*}} c"_ZN2ns8nskernelEv\00"
// CHECK: @[[TKERN:[0-9]*]] = {{.*}} c"_Z10kernelfuncIiEvv\00"

// Non-template kernel stub functions

// HIP: define{{.*}}@[[CSTUB]]
// CUDA: define{{.*}}@[[CSTUB:__device_stub__ckernel]]
// HIP: call{{.*}}@hipLaunchByPtr{{.*}}@[[HCKERN]]
// CUDA: call{{.*}}@cudaLaunch{{.*}}@[[CSTUB]]
// CUDA: store volatile i8 1, ptr @[[HCKERN]], align 1
// CHECK: ret void

// HIP: define{{.*}}@[[NSSTUB]]
// CUDA: define{{.*}}@[[NSSTUB:"\?__device_stub__nskernel@ns@@YAXXZ"]]
// HIP: call{{.*}}@hipLaunchByPtr{{.*}}@[[HNSKERN]]
// CUDA: call{{.*}}@cudaLaunch{{.*}}@[[NSSTUB]]
// CUDA: store volatile i8 1, ptr @[[HNSKERN]], align 1
// CHECK: ret void

// Check kernel stub is called for triple chevron.

// CHECK-LABEL: define{{.*}}@fun1()
// CHECK: call void @[[CSTUB]]()
// CHECK: call void @[[NSSTUB]]()
// HIP: call void @[[TSTUB]]()
// CUDA: call void @[[TSTUB:"\?\?\$__device_stub__kernelfunc@H@@YAXXZ.*"]]()
// GNU: call void @[[DSTUB:_Z26__device_stub__kernel_declv]]()
// GNU: call void @[[TDSTUB:_Z35__device_stub__template_kernel_declIiEvT_]](
// MSVC: call void @[[DSTUB:"\?__device_stub__kernel_decl@@YAXXZ"]]()
// MSVC: call void @[[TDSTUB:"\?\?\$__device_stub__template_kernel_decl@H@@YAXH@Z"]](

extern "C" void fun1(void) {
  ckernel<<<1, 1>>>();
  ns::nskernel<<<1, 1>>>();
  kernelfunc<int><<<1, 1>>>();
  kernel_decl<<<1, 1>>>();
  template_kernel_decl<<<1, 1>>>(1);
}

// Template kernel stub functions

// CHECK: define{{.*}}@[[TSTUB]]
// HIP: call{{.*}}@hipLaunchByPtr{{.*}}@[[HTKERN]]
// CUDA: call{{.*}}@cudaLaunch{{.*}}@[[TSTUB]]
// CUDA: store volatile i8 1, ptr @[[HTKERN]], align 1
// CHECK: ret void

// Check declaration of stub function for external kernel.

// CHECK: declare{{.*}}@[[DSTUB]]
// CHECK: declare{{.*}}@[[TDSTUB]]

// Check kernel handle is used for passing the kernel as a function pointer.

// CHECK-LABEL: define{{.*}}@fun2()
// HIP: call void @launch({{.*}}[[HCKERN]]
// HIP: call void @launch({{.*}}[[HNSKERN]]
// HIP: call void @launch({{.*}}[[HTKERN]]
// HIP: call void @launch({{.*}}[[HDKERN]]
// HIP: call void @launch({{.*}}[[HTDKERN]]
extern "C" void fun2() {
  launch((void *)ckernel);
  launch((void *)ns::nskernel);
  launch((void *)kernelfunc<int>);
  launch((void *)kernel_decl);
  launch((void *)template_kernel_decl<int>);
}

// Check kernel handle is used for assigning a kernel to a function pointer.

// CHECK-LABEL: define{{.*}}@fun3()
// HIP:  store ptr @[[HCKERN]], ptr @kernel_ptr, align 8
// HIP:  store ptr @[[HCKERN]], ptr @kernel_ptr, align 8
// HIP:  store ptr @[[HCKERN]], ptr @void_ptr, align 8
// HIP:  store ptr @[[HCKERN]], ptr @void_ptr, align 8
extern "C" void fun3() {
  kernel_ptr = ckernel;
  kernel_ptr = &ckernel;
  void_ptr = (void *)ckernel;
  void_ptr = (void *)&ckernel;
}

// Check kernel stub is loaded from kernel handle when function pointer is
// used with triple chevron.

// CHECK-LABEL: define{{.*}}@fun4()
// HIP:  store ptr @[[HCKERN]], ptr @kernel_ptr
// HIP:  call noundef i32 @{{.*hipConfigureCall}}
// HIP:  %[[HANDLE:.*]] = load ptr, ptr @kernel_ptr, align 8
// HIP:  %[[STUB:.*]] = load ptr, ptr %[[HANDLE]], align 8
// HIP:  call void %[[STUB]]()
extern "C" void fun4() {
  kernel_ptr = ckernel;
  kernel_ptr<<<1,1>>>();
}

// Check kernel handle is passed to a function.

// CHECK-LABEL: define{{.*}}@fun5()
// HIP:  store ptr @[[HCKERN]], ptr @kernel_ptr
// HIP:  %[[HANDLE:.*]] = load ptr, ptr @kernel_ptr, align 8
// HIP:  call void @launch(ptr noundef %[[HANDLE]])
extern "C" void fun5() {
  kernel_ptr = ckernel;
  launch((void *)kernel_ptr);
}

// Check kernel handle is registered.

// HIP-LABEL: define{{.*}}@__hip_register_globals
// HIP: call{{.*}}@__hipRegisterFunction{{.*}}@[[HCKERN]]{{.*}}@[[CKERN]]
// HIP: call{{.*}}@__hipRegisterFunction{{.*}}@[[HNSKERN]]{{.*}}@[[NSKERN]]
// HIP: call{{.*}}@__hipRegisterFunction{{.*}}@[[HTKERN]]{{.*}}@[[TKERN]]
// NEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}__device_stub
// NEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}kernel_decl
// NEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}template_kernel_decl
