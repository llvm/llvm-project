// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-cir -o - %s | FileCheck --check-prefix=CIR-DEVICE %s
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=LLVM-DEVICE %s
// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=OGCG-DEVICE %s

// RUN: echo -n "GPU binary would be here." > %t
// RUN: %clang_cc1 -fclangir -std=c++11 -triple x86_64-unknown-linux-gnu \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t \
// RUN:   -emit-cir -o - %s | FileCheck --check-prefix=CIR-HOST %s
// RUN: %clang_cc1 -fclangir -std=c++11 -triple x86_64-unknown-linux-gnu \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=LLVM-HOST %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu \
// RUN:   -target-sdk-version=12.3 -fcuda-include-gpubinary %t \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=OGCG-HOST %s

struct surfaceReference {
  int desc;
};

template <typename T, int dim = 1>
struct __attribute__((device_builtin_surface_type)) surface
    : public surfaceReference {};

template <int dim>
struct __attribute__((device_builtin_surface_type)) surface<void, dim>
    : public surfaceReference {};

surface<void, 2> surf;

//===----------------------------------------------------------------------===//
// Device-side checks
//===----------------------------------------------------------------------===//

// CIR-DEVICE: cir.global external target_address_space(1) @surf = #cir.undef : !cir.cuda_surface

// CIR now matches OG CodeGen and emits undef for CUDA shadow variables.
// LLVM-DEVICE: @surf ={{.*}} addrspace(1) externally_initialized global i64 undef
// OGCG-DEVICE: @surf ={{.*}} addrspace(1) externally_initialized global i64 undef

//===----------------------------------------------------------------------===//
// Host-side checks
//===----------------------------------------------------------------------===//

// Check the CUDA surface registration runtime declaration.
// CIR-HOST: cir.func private @__cudaRegisterSurface(!cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>, !s32i, !s32i)

// Check that __cuda_register_globals registers the surface using the host
// shadow, device-side name, surface type, and extern flag.
// CIR-HOST-LABEL: cir.func internal private @__cuda_register_globals
// CIR-HOST-SAME: (%[[FATBIN:.*]]: !cir.ptr<!cir.ptr<!void>>
// CIR-HOST: %[[NAME_RAW:.*]] = cir.get_global @".strsurf"
// CIR-HOST-NEXT: %[[NAME:.*]] = cir.cast bitcast %[[NAME_RAW]]
// CIR-HOST-NEXT: %[[HOST_RAW:.*]] = cir.get_global @surf
// CIR-HOST-NEXT: %[[HOST:.*]] = cir.cast bitcast %[[HOST_RAW]]
// CIR-HOST-NEXT: %[[EXTERN:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[SURFACE_TYPE:.*]] = cir.const #cir.int<2> : !s32i
// CIR-HOST-NEXT: cir.call @__cudaRegisterSurface(%[[FATBIN]], %[[HOST]], %[[NAME]], %[[NAME]], %[[SURFACE_TYPE]], %[[EXTERN]])

// Check that the host-side shadow carries the surface registration metadata,
// including the surface type extracted from surface<void, 2>.
// CIR-HOST: cir.global{{.*}} @surf = {{.*}}cu.var_registration = #cir.cu.var_registration<surf, Surface, surface_type = 2>

// Check CIR-lowered LLVM registration.
// LLVM-HOST-LABEL: define internal void @__cuda_register_globals
// LLVM-HOST-SAME: (ptr %[[FATBIN:.*]])
// LLVM-HOST: call void @__cudaRegisterSurface(ptr %[[FATBIN]], ptr @surf, ptr @[[NAME:.*]], ptr @[[NAME]], i32 2, i32 0)

// Check parity with original CodeGen.
// OGCG-HOST-LABEL: define internal void @__cuda_register_globals
// OGCG-HOST-SAME: (ptr %[[FATBIN:.*]])
// OGCG-HOST: call void @__cudaRegisterSurface(ptr %[[FATBIN]], ptr @surf, ptr @[[NAME:.*]], ptr @[[NAME]], i32 2, i32 0)
