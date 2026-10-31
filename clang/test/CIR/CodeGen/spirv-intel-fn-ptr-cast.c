// RUN: %clang_cc1 -triple spirv64-intel -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spirv64-intel -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple spirv64-intel -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// CIR-LABEL: cir.func{{.*}} @fp_to_void_ptr
// CIR: %[[FP:.*]] = cir.load{{.*}} : {{.*}}, !cir.ptr<!cir.func<()>, target_address_space(9)>
// CIR: %[[AS:.*]] = cir.cast address_space %[[FP]] : !cir.ptr<!cir.func<()>, target_address_space(9)> -> !cir.ptr<!cir.func<()>, target_address_space(4)>
// CIR: cir.cast bitcast %[[AS]] : !cir.ptr<!cir.func<()>, target_address_space(4)> -> !cir.ptr<!void, target_address_space(4)>

// LLVM-LABEL: define{{.*}} ptr addrspace(4) @fp_to_void_ptr
// LLVM: %[[FP:.*]] = load ptr addrspace(9), ptr addrspace(4)
// LLVM: addrspacecast ptr addrspace(9) %[[FP]] to ptr addrspace(4)

void *fp_to_void_ptr(void (*fp)(void)) { return (void *)fp; }

// CIR-LABEL: cir.func{{.*}} @void_ptr_to_fp
// CIR: %[[P:.*]] = cir.load{{.*}} : {{.*}}, !cir.ptr<!void, target_address_space(4)>
// CIR: %[[AS:.*]] = cir.cast address_space %[[P]] : !cir.ptr<!void, target_address_space(4)> -> !cir.ptr<!void, target_address_space(9)>
// CIR: cir.cast bitcast %[[AS]] : !cir.ptr<!void, target_address_space(9)> -> !cir.ptr<!cir.func<()>, target_address_space(9)>

// LLVM-LABEL: define{{.*}} ptr addrspace(9) @void_ptr_to_fp
// LLVM: %[[P:.*]] = load ptr addrspace(4), ptr addrspace(4)
// LLVM: addrspacecast ptr addrspace(4) %[[P]] to ptr addrspace(9)

void (*void_ptr_to_fp(void *p))(void) { return (void (*)(void))p; }
