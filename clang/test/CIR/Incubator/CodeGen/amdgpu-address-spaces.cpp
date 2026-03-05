// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Test address space handling for AMDGPU target in C++ mode (non-OpenCL/HIP).
// This exercises getGlobalVarAddressSpace.

// Test default address space for globals without explicit AS.
// For AMDGPU in non-OpenCL/HIP mode, globals default to AS 1 (global).
int globalVar = 123;

// CIR-DAG: cir.global external lang_address_space(offload_global) @globalVar = #cir.int<123> : !s32i
// LLVM-DAG: @globalVar = addrspace(1) global i32 123, align 4
// OGCG-DAG: @globalVar = addrspace(1) global i32 123, align 4

// Test non-const global array goes to global AS.
int globalArray[4] = {1, 2, 3, 4};

// CIR-DAG: cir.global external lang_address_space(offload_global) @globalArray = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 4>
// LLVM-DAG: @globalArray = addrspace(1) global [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4
// OGCG-DAG: @globalArray = addrspace(1) global [4 x i32] [i32 1, i32 2, i32 3, i32 4], align 4

// Test static global goes to global AS.
static int staticGlobal = 555;

// CIR-DAG: cir.global "private" internal{{.*}}lang_address_space(offload_global) @_ZL12staticGlobal = #cir.int<555> : !s32i
// LLVM-DAG: @_ZL12staticGlobal = internal addrspace(1) global i32 555, align 4
// OGCG-DAG: @_ZL12staticGlobal = internal addrspace(1) global i32 555, align 4

// Test constant initialization promotion to AS 4 (constant).
// Use extern to force emission since const globals are otherwise optimized away.
extern const int constGlobal = 456;

// CIR-DAG: cir.global constant external target_address_space(4) @constGlobal = #cir.int<456> : !s32i
// LLVM-DAG: @constGlobal = addrspace(4) constant i32 456, align 4
// OGCG-DAG: @constGlobal = addrspace(4) constant i32 456, align 4

// Test extern const array goes to constant AS.
extern const int constArray[3] = {10, 20, 30};

// CIR-DAG: cir.global constant external target_address_space(4) @constArray = #cir.const_array<[#cir.int<10> : !s32i, #cir.int<20> : !s32i, #cir.int<30> : !s32i]> : !cir.array<!s32i x 3>
// LLVM-DAG: @constArray = addrspace(4) constant [3 x i32] [i32 10, i32 20, i32 30], align 4
// OGCG-DAG: @constArray = addrspace(4) constant [3 x i32] [i32 10, i32 20, i32 30], align 4

// Use the static variable to ensure it's emitted.
int getStaticGlobal() { return staticGlobal; }
