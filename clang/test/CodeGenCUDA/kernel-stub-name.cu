// RUN: echo "GPU binary would be here" > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck %s

#include "Inputs/cuda.h"

// Kernel handles

// CHECK: @[[HCKERN:ckernel]] = constant void ()* @__device_stub__ckernel, align 8
// CHECK: @[[HNSKERN:_ZN2ns8nskernelEv]] = constant void ()* @_ZN2ns23__device_stub__nskernelEv, align 8
// CHECK: @[[HTKERN:_Z10kernelfuncIiEvv]] = linkonce_odr constant void ()* @_Z25__device_stub__kernelfuncIiEvv, align 8
// CHECK: @[[HDKERN:_Z11kernel_declv]] = external constant void ()*, align 8

extern "C" __global__ void ckernel() {}

namespace ns {
__global__ void nskernel() {}
} // namespace ns

template<class T>
__global__ void kernelfunc() {}

__global__ void kernel_decl();

void (*kernel_ptr)();
void *void_ptr;

void launch(void *kern);

// Device side kernel names

// CHECK: @[[CKERN:[0-9]*]] = {{.*}} c"ckernel\00"
// CHECK: @[[NSKERN:[0-9]*]] = {{.*}} c"_ZN2ns8nskernelEv\00"
// CHECK: @[[TKERN:[0-9]*]] = {{.*}} c"_Z10kernelfuncIiEvv\00"

// Non-template kernel stub functions

// CHECK: define{{.*}}@[[CSTUB:__device_stub__ckernel]]
// CHECK: call{{.*}}@hipLaunchByPtr{{.*}}@[[HCKERN]]
// CHECK: define{{.*}}@[[NSSTUB:_ZN2ns23__device_stub__nskernelEv]]
// CHECK: call{{.*}}@hipLaunchByPtr{{.*}}@[[HNSKERN]]


// Check kernel stub is used for triple chevron

// CHECK-LABEL: define{{.*}}@_Z4fun1v()
// CHECK: call void @[[CSTUB]]()
// CHECK: call void @[[NSSTUB]]()
// CHECK: call void @[[TSTUB:_Z25__device_stub__kernelfuncIiEvv]]()
// CHECK: call void @[[DSTUB:_Z26__device_stub__kernel_declv]]()

void fun1(void) {
  ckernel<<<1, 1>>>();
  ns::nskernel<<<1, 1>>>();
  kernelfunc<int><<<1, 1>>>();
  kernel_decl<<<1, 1>>>();
}

// Template kernel stub functions

// CHECK: define{{.*}}@[[TSTUB]]
// CHECK: call{{.*}}@hipLaunchByPtr{{.*}}@[[HTKERN]]

// Check declaration of stub function for external kernel.

// CHECK: declare{{.*}}@[[DSTUB]]

// Check kernel handle is used for passing the kernel as a function pointer

// CHECK-LABEL: define{{.*}}@_Z4fun2v()
// CHECK: call void @_Z6launchPv({{.*}}[[HCKERN]]
// CHECK: call void @_Z6launchPv({{.*}}[[HNSKERN]]
// CHECK: call void @_Z6launchPv({{.*}}[[HTKERN]]
// CHECK: call void @_Z6launchPv({{.*}}[[HDKERN]]
void fun2() {
  launch((void *)ckernel);
  launch((void *)ns::nskernel);
  launch((void *)kernelfunc<int>);
  launch((void *)kernel_decl);
}

// Check kernel handle is used for assigning a kernel to a function pointer

// CHECK-LABEL: define{{.*}}@_Z4fun3v()
// CHECK:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr, align 8
// CHECK:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr, align 8
// CHECK:  store i8* bitcast (void ()** @[[HCKERN]] to i8*), i8** @void_ptr, align 8
// CHECK:  store i8* bitcast (void ()** @[[HCKERN]] to i8*), i8** @void_ptr, align 8
void fun3() {
  kernel_ptr = ckernel;
  kernel_ptr = &ckernel;
  void_ptr = (void *)ckernel;
  void_ptr = (void *)&ckernel;
}

// Check kernel stub is loaded from kernel handle when function pointer is
// used with triple chevron

// CHECK-LABEL: define{{.*}}@_Z4fun4v()
// CHECK:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr
// CHECK:  call i32 @_Z16hipConfigureCall4dim3S_mP9hipStream
// CHECK:  %[[HANDLE:.*]] = load void ()*, void ()** @kernel_ptr, align 8
// CHECK:  %[[CAST:.*]] = bitcast void ()* %[[HANDLE]] to void ()**
// CHECK:  %[[STUB:.*]] = load void ()*, void ()** %[[CAST]], align 8
// CHECK:  call void %[[STUB]]()
void fun4() {
  kernel_ptr = ckernel;
  kernel_ptr<<<1,1>>>();
}

// Check kernel handle is passed to a function

// CHECK-LABEL: define{{.*}}@_Z4fun5v()
// CHECK:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr
// CHECK:  %[[HANDLE:.*]] = load void ()*, void ()** @kernel_ptr, align 8
// CHECK:  %[[CAST:.*]] = bitcast void ()* %[[HANDLE]] to i8*
// CHECK:  call void @_Z6launchPv(i8* %[[CAST]])
void fun5() {
  kernel_ptr = ckernel;
  launch((void *)kernel_ptr);
}

// CHECK-LABEL: define{{.*}}@__hip_register_globals
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[HCKERN]]{{.*}}@[[CKERN]]
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[HNSKERN]]{{.*}}@[[NSKERN]]
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[HTKERN]]{{.*}}@[[TKERN]]
// CHECK-NOT: call{{.*}}@__hipRegisterFunction{{.*}}@[[HDKERN]]{{.*}}@[[DKERN]]
