// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR-SPV %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM-SPV %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=OGCG-SPV %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR-GCN %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM-GCN %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=OGCG-GCN %s

// String literals are emitted in the target's constant address space for
// globals (CrossWorkgroup on SPIR-V, constant on AMDGCN) and are referenced
// through an addrspacecast to the default address space.

#define __device__ __attribute__((device))

__device__ void take(const char *);

// CIR-SPV: cir.global "private" constant cir_private dso_local target_address_space(1) @".str" = #cir.const_array<"hi" : !cir.array<!s8i x 2>, trailing_zeros> : !cir.array<!s8i x 3>
// CIR-SPV: cir.global "private" constant cir_private dso_local target_address_space(1) @".str.1" = #cir.const_array<"glob" : !cir.array<!s8i x 4>, trailing_zeros> : !cir.array<!s8i x 5>
// CIR-SPV: cir.global external target_address_space(1) @gp = #cir.global_view<@".str.1"> : !cir.ptr<!s8i, target_address_space(4)>

// CIR-GCN: cir.global "private" constant cir_private dso_local target_address_space(4) @".str" = #cir.const_array<"hi" : !cir.array<!s8i x 2>, trailing_zeros> : !cir.array<!s8i x 3>
// CIR-GCN: cir.global "private" constant cir_private dso_local target_address_space(4) @".str.1" = #cir.const_array<"glob" : !cir.array<!s8i x 4>, trailing_zeros> : !cir.array<!s8i x 5>
// CIR-GCN: cir.global external target_address_space(1) @gp = #cir.global_view<@".str.1"> : !cir.ptr<!s8i>

// LLVM-SPV: @.str = private addrspace(1) constant [3 x i8] c"hi\00"
// LLVM-SPV: @.str.1 = private addrspace(1) constant [5 x i8] c"glob\00"
// LLVM-SPV: @gp = addrspace(1) externally_initialized global ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str.1 to ptr addrspace(4))

// OGCG-SPV: @.str = private unnamed_addr addrspace(1) constant [3 x i8] c"hi\00"
// OGCG-SPV: @.str.1 = private unnamed_addr addrspace(1) constant [5 x i8] c"glob\00"
// OGCG-SPV: @gp = addrspace(1) externally_initialized global ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str.1 to ptr addrspace(4))

// LLVM-GCN: @.str = private addrspace(4) constant [3 x i8] c"hi\00"
// LLVM-GCN: @.str.1 = private addrspace(4) constant [5 x i8] c"glob\00"
// LLVM-GCN: @gp = addrspace(1) externally_initialized global ptr addrspacecast (ptr addrspace(4) @.str.1 to ptr)

// OGCG-GCN: @.str = private unnamed_addr addrspace(4) constant [3 x i8] c"hi\00"
// OGCG-GCN: @.str.1 = private unnamed_addr addrspace(4) constant [5 x i8] c"glob\00"
// OGCG-GCN: @gp = addrspace(1) externally_initialized global ptr addrspacecast (ptr addrspace(4) @.str.1 to ptr)

// CIR-SPV-LABEL: cir.func{{.*}} @_Z9call_takev
// CIR-SPV: %[[G:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(1)>
// CIR-SPV: %[[C:.*]] = cir.cast address_space %[[G]] : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(1)> -> !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)>
// CIR-SPV: %[[D:.*]] = cir.cast array_to_ptrdecay %[[C]] : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)> -> !cir.ptr<!s8i, target_address_space(4)>
// CIR-SPV: cir.call @_Z4takePKc(%[[D]])

// CIR-GCN-LABEL: cir.func{{.*}} @_Z9call_takev
// CIR-GCN: %[[G:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)>
// CIR-GCN: %[[C:.*]] = cir.cast address_space %[[G]] : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)> -> !cir.ptr<!cir.array<!s8i x 3>>
// CIR-GCN: %[[D:.*]] = cir.cast array_to_ptrdecay %[[C]] : !cir.ptr<!cir.array<!s8i x 3>> -> !cir.ptr<!s8i>
// CIR-GCN: cir.call @_Z4takePKc(%[[D]])

// LLVM-SPV-LABEL: define{{.*}} void @_Z9call_takev
// LLVM-SPV: call{{.*}} void @_Z4takePKc(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)))

// OGCG-SPV-LABEL: define{{.*}} void @_Z9call_takev
// OGCG-SPV: call{{.*}} void @_Z4takePKc(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)))

// LLVM-GCN-LABEL: define{{.*}} void @_Z9call_takev
// LLVM-GCN: call void @_Z4takePKc(ptr noundef addrspacecast (ptr addrspace(4) @.str to ptr))

// OGCG-GCN-LABEL: define{{.*}} void @_Z9call_takev
// OGCG-GCN: call void @_Z4takePKc(ptr noundef addrspacecast (ptr addrspace(4) @.str to ptr))
__device__ void call_take() { take("hi"); }

// A second use of the same literal reuses the cached global.

// CIR-SPV-LABEL: cir.func{{.*}} @_Z3retv
// CIR-SPV: %[[G:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(1)>
// CIR-SPV: cir.cast address_space %[[G]] : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(1)> -> !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)>

// CIR-GCN-LABEL: cir.func{{.*}} @_Z3retv
// CIR-GCN: %[[G:.*]] = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)>
// CIR-GCN: cir.cast address_space %[[G]] : !cir.ptr<!cir.array<!s8i x 3>, target_address_space(4)> -> !cir.ptr<!cir.array<!s8i x 3>>

// LLVM-SPV-LABEL: define{{.*}} ptr addrspace(4) @_Z3retv
// LLVM-SPV: store ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4))

// OGCG-SPV-LABEL: define{{.*}} ptr addrspace(4) @_Z3retv
// OGCG-SPV: ret ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4))

// LLVM-GCN-LABEL: define{{.*}} ptr @_Z3retv
// LLVM-GCN: store ptr addrspacecast (ptr addrspace(4) @.str to ptr)

// OGCG-GCN-LABEL: define{{.*}} ptr @_Z3retv
// OGCG-GCN: ret ptr addrspacecast (ptr addrspace(4) @.str to ptr)
__device__ const char *ret() { return "hi"; }

// Initializing a global goes through constant emission instead.
__device__ const char *gp = "glob";
