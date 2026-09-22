#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-cxxabi-lowering %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=CIR-BEFORE
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=LLVM,OGCG

// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-cxxabi-lowering %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefix=CIR-BEFORE
// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple amdgpu9.00-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=LLVM,OGCG

namespace std {
class type_info {
public:
  virtual ~type_info();
};
} // namespace std

struct B {
  __device__ virtual ~B();
};
struct D : B {};

__device__ D *ptr_cast(B *b) { return dynamic_cast<D *>(b); }

// CIR-BEFORE-LABEL: cir.func {{.*}} @_Z8ptr_castP1B(
// CIR-BEFORE:         cir.dyn_cast ptr %{{.+}} : !cir.ptr<!rec_B> -> !cir.ptr<!rec_D> #cir.dyn_cast_info<runtime_func = @__dynamic_cast, bad_cast_func = @__cxa_bad_cast, offset_hint = #cir.int<0> : !s64i>

// CIR-LABEL: cir.func {{.*}} @_Z8ptr_castP1B(
// CIR:         %[[SRC_RTTI:.+]] = cir.const #cir.ptr<null> : !cir.ptr<!u8i>
// CIR:         %[[DEST_RTTI:.+]] = cir.const #cir.ptr<null> : !cir.ptr<!u8i>
// CIR:         cir.call @__dynamic_cast(%{{.+}}, %[[SRC_RTTI]], %[[DEST_RTTI]], %{{.+}})

// LLVM-LABEL: define {{.*}} ptr @_Z8ptr_castP1B(
// LLVM:         call ptr @__dynamic_cast(ptr %{{.+}}, ptr {{.*}}null, ptr {{.*}}null, i64 0)

__device__ D &ref_cast(B &b) { return dynamic_cast<D &>(b); }

// CIR-BEFORE-LABEL: cir.func {{.*}} @_Z8ref_castR1B(
// CIR-BEFORE:         cir.dyn_cast ref %{{.+}} : !cir.ptr<!rec_B> -> !cir.ptr<!rec_D> #cir.dyn_cast_info<runtime_func = @__dynamic_cast, bad_cast_func = @__cxa_bad_cast, offset_hint = #cir.int<0> : !s64i>

// CIR-LABEL: cir.func {{.*}} @_Z8ref_castR1B(
// CIR:         %[[SRC_RTTI:.+]] = cir.const #cir.ptr<null> : !cir.ptr<!u8i>
// CIR:         %[[DEST_RTTI:.+]] = cir.const #cir.ptr<null> : !cir.ptr<!u8i>
// CIR:         cir.call @__dynamic_cast(%{{.+}}, %[[SRC_RTTI]], %[[DEST_RTTI]], %{{.+}})
// CIR:         cir.call @__cxa_bad_cast()

// LLVM-LABEL: define {{.*}} ptr @_Z8ref_castR1B(
// LLVM:         call ptr @__dynamic_cast(ptr %{{.+}}, ptr {{.*}}null, ptr {{.*}}null, i64 0)
// LLVM:         call void @__cxa_bad_cast()

__device__ const std::type_info &static_typeid() { return typeid(D); }

// CIR-LABEL: cir.func {{.*}} @_Z13static_typeidv(
// CIR:         %[[RTTI:.+]] = cir.const #cir.ptr<null> : !cir.ptr<!u8i>
// CIR:         cir.cast bitcast %[[RTTI]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_std3A3Atype_info>

// LLVM-LABEL: define {{.*}} ptr @_Z13static_typeidv(
// LLVMCIR:      store ptr null, ptr {{(addrspace\(5\) )?}}[[RET:%.+]], align 8
// LLVMCIR:      [[RES:%.+]] = load ptr, ptr {{(addrspace\(5\) )?}}[[RET]], align 8
// LLVMCIR:      ret ptr [[RES]]
// OGCG:         ret ptr {{null|addrspacecast \(ptr addrspace\(1\) null to ptr\)}}
