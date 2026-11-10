// RTTI is not emitted when compiling for a GPU offload device (SYCL, OpenMP,
// CUDA or HIP), but is emitted for the host. Without RTTI, the vtable's RTTI
// slot is null and no type info (_ZTI) is emitted.

// SYCL device on SPIR.
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-sycl.cir
// RUN: FileCheck --check-prefix=CIR-DEV --implicit-check-not=_ZTI --input-file=%t-sycl.cir %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-sycl-cir.ll
// RUN: FileCheck --check-prefix=LLVM-DEV --implicit-check-not=_ZTI --input-file=%t-sycl-cir.ll %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-sycl.ll
// RUN: FileCheck --check-prefix=LLVM-DEV --implicit-check-not=_ZTI --input-file=%t-sycl.ll %s

// OpenMP device on AMDGPU.
// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-omp.cir
// RUN: FileCheck --check-prefix=CIR-DEV --implicit-check-not=_ZTI --input-file=%t-omp.cir %s
// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-omp-cir.ll
// RUN: FileCheck --check-prefix=LLVM-DEV --implicit-check-not=_ZTI --input-file=%t-omp-cir.ll %s
// RUN: %clang_cc1 -fopenmp -fopenmp-is-target-device -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-omp.ll
// RUN: FileCheck --check-prefix=LLVM-DEV --implicit-check-not=_ZTI --input-file=%t-omp.ll %s

// CUDA and HIP devices (unchanged behavior).
// RUN: %clang_cc1 -x cuda -fcuda-is-device -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-cuda.cir
// RUN: FileCheck --check-prefix=CIR-DEV --implicit-check-not=_ZTI --input-file=%t-cuda.cir %s
// RUN: %clang_cc1 -x hip -fcuda-is-device -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-hip.cir
// RUN: FileCheck --check-prefix=CIR-DEV --implicit-check-not=_ZTI --input-file=%t-hip.cir %s

// SYCL host keeps RTTI.
// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-host.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t-host.cir %s
// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-host-cir.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t-host-cir.ll %s
// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-host.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t-host.ll %s

#if defined(__CUDA__) || defined(__HIP__)
#define DEVICE __attribute__((device))
#else
#define DEVICE
#endif

#ifdef _OPENMP
#pragma omp declare target
#endif

struct Key {
  DEVICE virtual void f();
};

#ifdef __SYCL_DEVICE_ONLY__
[[clang::sycl_external]]
#endif
DEVICE void Key::f() {}

#ifdef _OPENMP
#pragma omp end declare target
#endif

// CIR-DEV: cir.global {{.*}} @_ZTV3Key = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZN3Key1fEv> : !cir.ptr<!u8i>]>
// LLVM-DEV: @_ZTV3Key = {{.*}}null, ptr{{.*}} null, ptr{{.*}} @_ZN3Key1fEv

// CIR-HOST-DAG: cir.global {{.*}} @_ZTV3Key = #cir.vtable<{#cir.const_array<[#cir.ptr<null> : !cir.ptr<!u8i>, #cir.global_view<@_ZTI3Key> : !cir.ptr<!u8i>, #cir.global_view<@_ZN3Key1fEv> : !cir.ptr<!u8i>]>
// CIR-HOST-DAG: cir.global constant external @_ZTI3Key = #cir.typeinfo<
// LLVM-HOST-DAG: @_ZTV3Key = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI3Key, ptr @_ZN3Key1fEv] }
// LLVM-HOST-DAG: @_ZTI3Key = constant
