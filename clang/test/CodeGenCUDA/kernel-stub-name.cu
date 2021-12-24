// RUN: echo "GPU binary would be here" > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -check-prefixes=CHECK,GNU %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -check-prefix=NEG %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm %s \
// RUN:     -aux-triple amdgcn-amd-amdhsa -fcuda-include-gpubinary \
// RUN:     %t -o - -x hip\
// RUN:   | FileCheck -check-prefixes=CHECK,MSVC %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm %s \
// RUN:     -aux-triple amdgcn-amd-amdhsa -fcuda-include-gpubinary \
// RUN:     %t -o - -x hip\
// RUN:   | FileCheck -check-prefix=NEG %s

#include "Inputs/cuda.h"

// Check kernel handles are emitted for non-MSVC target but not for MSVC target.

// GNU: @[[HCKERN:ckernel]] = constant void ()* @[[CSTUB:__device_stub__ckernel]], align 8
// GNU: @[[HNSKERN:_ZN2ns8nskernelEv]] = constant void ()* @[[NSSTUB:_ZN2ns23__device_stub__nskernelEv]], align 8
// GNU: @[[HTKERN:_Z10kernelfuncIiEvv]] = linkonce_odr constant void ()* @[[TSTUB:_Z25__device_stub__kernelfuncIiEvv]], comdat, align 8
// GNU: @[[HDKERN:_Z11kernel_declv]] = external constant void ()*, align 8

// MSVC: @[[HCKERN:ckernel]] = dso_local constant void ()* @[[CSTUB:__device_stub__ckernel]], align 8
// MSVC: @[[HNSKERN:"\?nskernel@ns@@YAXXZ.*"]] = dso_local constant void ()* @[[NSSTUB:"\?__device_stub__nskernel@ns@@YAXXZ"]], align 8
// MSVC: @[[HTKERN:"\?\?\$kernelfunc@H@@YAXXZ.*"]] = linkonce_odr dso_local constant void ()* @[[TSTUB:"\?\?\$__device_stub__kernelfunc@H@@YAXXZ.*"]], comdat, align 8
// MSVC: @[[HDKERN:"\?kernel_decl@@YAXXZ.*"]] = external dso_local constant void ()*, align 8

extern "C" __global__ void ckernel() {}

namespace ns {
__global__ void nskernel() {}
} // namespace ns

template<class T>
__global__ void kernelfunc() {}

__global__ void kernel_decl();

extern "C" void (*kernel_ptr)();
extern "C" void *void_ptr;

extern "C" void launch(void *kern);

// Device side kernel names

// CHECK: @[[CKERN:[0-9]*]] = {{.*}} c"ckernel\00"
// CHECK: @[[NSKERN:[0-9]*]] = {{.*}} c"_ZN2ns8nskernelEv\00"
// CHECK: @[[TKERN:[0-9]*]] = {{.*}} c"_Z10kernelfuncIiEvv\00"

// Non-template kernel stub functions

// CHECK: define{{.*}}@[[CSTUB]]
// CHECK: call{{.*}}@hipLaunchByPtr{{.*}}@[[HCKERN]]

// CHECK: define{{.*}}@[[NSSTUB]]
// CHECK: call{{.*}}@hipLaunchByPtr{{.*}}@[[HNSKERN]]

// Check kernel stub is called for triple chevron.

// CHECK-LABEL: define{{.*}}@fun1()
// CHECK: call void @[[CSTUB]]()
// CHECK: call void @[[NSSTUB]]()
// CHECK: call void @[[TSTUB]]()
// GNU: call void @[[DSTUB:_Z26__device_stub__kernel_declv]]()
// MSVC: call void @[[DSTUB:"\?__device_stub__kernel_decl@@YAXXZ"]]()

extern "C" void fun1(void) {
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

// Check kernel handle is used for passing the kernel as a function pointer.

// CHECK-LABEL: define{{.*}}@fun2()
// CHECK: call void @launch({{.*}}[[HCKERN]]
// CHECK: call void @launch({{.*}}[[HNSKERN]]
// CHECK: call void @launch({{.*}}[[HTKERN]]
// CHECK: call void @launch({{.*}}[[HDKERN]]
extern "C" void fun2() {
  launch((void *)ckernel);
  launch((void *)ns::nskernel);
  launch((void *)kernelfunc<int>);
  launch((void *)kernel_decl);
}

// Check kernel handle is used for assigning a kernel to a function pointer.

// CHECK-LABEL: define{{.*}}@fun3()
// CHECK:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr, align 8
// CHECK:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr, align 8
// CHECK:  store i8* bitcast (void ()** @[[HCKERN]] to i8*), i8** @void_ptr, align 8
// CHECK:  store i8* bitcast (void ()** @[[HCKERN]] to i8*), i8** @void_ptr, align 8
extern "C" void fun3() {
  kernel_ptr = ckernel;
  kernel_ptr = &ckernel;
  void_ptr = (void *)ckernel;
  void_ptr = (void *)&ckernel;
}

// Check kernel stub is loaded from kernel handle when function pointer is
// used with triple chevron.

// CHECK-LABEL: define{{.*}}@fun4()
// CHECK:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr
// CHECK:  call i32 @{{.*hipConfigureCall}}
// CHECK:  %[[HANDLE:.*]] = load void ()*, void ()** @kernel_ptr, align 8
// CHECK:  %[[CAST:.*]] = bitcast void ()* %[[HANDLE]] to void ()**
// CHECK:  %[[STUB:.*]] = load void ()*, void ()** %[[CAST]], align 8
// CHECK:  call void %[[STUB]]()
extern "C" void fun4() {
  kernel_ptr = ckernel;
  kernel_ptr<<<1,1>>>();
}

// Check kernel handle is passed to a function.

// CHECK-LABEL: define{{.*}}@fun5()
// CHECK:  store void ()* bitcast (void ()** @[[HCKERN]] to void ()*), void ()** @kernel_ptr
// CHECK:  %[[HANDLE:.*]] = load void ()*, void ()** @kernel_ptr, align 8
// CHECK:  %[[CAST:.*]] = bitcast void ()* %[[HANDLE]] to i8*
// CHECK:  call void @launch(i8* %[[CAST]])
extern "C" void fun5() {
  kernel_ptr = ckernel;
  launch((void *)kernel_ptr);
}

// Check kernel handle is registered.

// CHECK-LABEL: define{{.*}}@__hip_register_globals
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[HCKERN]]{{.*}}@[[CKERN]]
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[HNSKERN]]{{.*}}@[[NSKERN]]
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[HTKERN]]{{.*}}@[[TKERN]]
// NEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}__device_stub
// NEG-NOT: call{{.*}}@__hipRegisterFunction{{.*}}kernel_decl
