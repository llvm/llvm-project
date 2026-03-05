// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -emit-cir -fclangir -o - %s | FileCheck %s --check-prefix=CIR
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O2 -emit-llvm -fclangir -o - %s | FileCheck %s --check-prefix=LLVM
// RUN: %clang -cc1 -triple spirv64-unknown-unknown -cl-std=CL2.0 -finclude-default-header -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=OG-LLVM

unsigned int test(local int* x) {
    return *(local unsigned int*)x;
}

// CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!s32i, lang_address_space(offload_local)> -> !cir.ptr<!u32i, lang_address_space(offload_local)>
// LLVM: load i32, ptr addrspace(3) %{{.*}}, align 4
// OG-LLVM: load i32, ptr addrspace(3) %{{.*}}, align 4

void atomic_flag_clear(volatile __global atomic_flag *obj, memory_order ord, memory_scope scp)
{
  __atomic_store_n((volatile __global uint *)obj, 0, ord);
}

// CIR: cir.cast bitcast %{{.*}} : !cir.ptr<!s32i, lang_address_space(offload_global)> -> !cir.ptr<!u32i, lang_address_space(offload_global)>
// LLVM-COUNT-3: store atomic volatile i32 0, ptr addrspace(1) 
// OG-LLVM-COUNT-3: store atomic volatile i32 0, ptr addrspace(1) 
