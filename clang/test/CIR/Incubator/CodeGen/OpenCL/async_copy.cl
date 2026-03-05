// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -emit-cir -o - %s -fclangir | FileCheck %s --check-prefix=CIR-SPIR
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -emit-llvm -o - %s -fclangir | FileCheck %s --check-prefix=LLVM-SPIR
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -emit-llvm -o - %s | FileCheck %s --check-prefix=OG-LLVM-SPIR

// RUN: %clang -cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -finclude-default-header -emit-cir -o - %s -fclangir | FileCheck %s --check-prefix=CIR-AMDGCN
// RUN: %clang -cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -finclude-default-header -emit-llvm -o - %s -fclangir | FileCheck %s --check-prefix=LLVM-AMDGCN
// RUN: %clang -cc1 -triple amdgcn-amd-amdhsa -cl-std=CL2.0 -finclude-default-header -emit-llvm -o - %s | FileCheck %s --check-prefix=OG-LLVM-AMDGCN


// Simple kernel using async_work_group_copy + wait_group_events

__kernel void test_async_copy(__global int *g_in, __local int *l_in, int size) {
    // int gid = get_global_id(0);

    // Trigger async copy: global to local
    // event_t e_in = 
    async_work_group_copy(
        l_in,                          // local destination
        g_in,// + gid * size,             // global source
        size,                          // number of elements
        (event_t)0                     // no dependency
    );

    // Wait for the async operation to complete
    // wait_group_events(1, &e_in);
}

// CIR-SPIR: cir.call @_Z21async_work_group_copyPU3AS3iPU3AS1Kim9ocl_event(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!cir.ptr<!s32i, lang_address_space(offload_local)>, !cir.ptr<!s32i, lang_address_space(offload_global)>, !u64i, !cir.opaque<"event">) -> !cir.opaque<"event">
// LLVM-SPIR: call spir_func target("spirv.Event") @_Z21async_work_group_copyPU3AS3iPU3AS1Kim9ocl_event(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}}, i64 %{{.*}}, target("spirv.Event") zeroinitializer)
// OG-LLVM-SPIR: call spir_func target("spirv.Event") @_Z21async_work_group_copyPU3AS3iPU3AS1Kim9ocl_event(ptr addrspace(3) noundef %{{.*}}, ptr addrspace(1) noundef %{{.*}}, i64 noundef %{{.*}}, target("spirv.Event") zeroinitializer

// CIR-AMDGCN: cir.call @_Z21async_work_group_copyPU3AS3iPU3AS1Kim9ocl_event(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!cir.ptr<!s32i, lang_address_space(offload_local)>, !cir.ptr<!s32i, lang_address_space(offload_global)>, !u64i, !cir.opaque<"event">) -> !cir.opaque<"event">
// LLVM-AMDGCN: call ptr @_Z21async_work_group_copyPU3AS3iPU3AS1Kim9ocl_event(ptr addrspace(3) %{{.*}}, ptr addrspace(1) %{{.*}}, i64 %{{.*}}, ptr null)
// OG-LLVM-AMDGCN: call ptr @_Z21async_work_group_copyPU3AS3iPU3AS1Kim9ocl_event(ptr addrspace(3) noundef %{{.*}}, ptr addrspace(1) noundef %{{.*}}, i64 noundef %{{.*}}, ptr null)