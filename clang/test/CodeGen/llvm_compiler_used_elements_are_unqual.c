// RUN: %clang_cc1 -x c -triple x86_64-- -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -x c -triple amdgcn-amd-amdhsa -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=AMDGCN
// RUN: %clang_cc1 -x c -triple spirv64-- -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=SPIRV
// RUN: %clang_cc1 -x c -triple spirv64-amd-amdhsa -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=SPIRV_AMD
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 -triple x86_64-- -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 -triple amdgcn-amd-amdhsa -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=AMDGCN
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 -triple spirv64-- -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=SPIRV_CL
// RUN: %clang_cc1 -x cl -cl-std=CL1.2 -triple spirv64-amd-amdhsa -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=SPIRV_AMD_CL
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 -triple x86_64-- -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=AMDGCN
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 -triple spirv64-- -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=SPIRV_CL
// RUN: %clang_cc1 -x cl -cl-std=CL2.0 -triple spirv64-amd-amdhsa -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=SPIRV_AMD_CL

#ifndef __OPENCL_C_VERSION__
#define __constant const
#endif

static __constant __attribute__((__used__)) int foo = 42;


// X86: @foo = internal constant i32 42
// X86: @llvm.compiler.used = appending global [2 x ptr] [ptr @foo, ptr @bar], section "llvm.metadata"
//
// AMDGCN: @foo = internal addrspace(4) constant i32 42
// AMDGCN: @llvm.compiler.used = appending addrspace(1) global [2 x ptr] [ptr addrspacecast (ptr addrspace(4) @foo to ptr), ptr @bar], section "llvm.metadata"
//
// SPIRV: @foo = internal constant i32 42
// SPIRV: @llvm.used = appending addrspace(1) global [2 x ptr] [ptr @foo, ptr @bar], section "llvm.metadata"
//
// SPIRV_CL: @foo = internal addrspace(2) constant i32 42
// SPIRV_CL: @llvm.used = appending addrspace(1) global [2 x ptr] [ptr addrspacecast (ptr addrspace(2) @foo to ptr), ptr @bar], section "llvm.metadata"
//
// SPIRV_AMD: @foo = internal addrspace(1) constant i32 42
// SPIRV_AMD: @llvm.used = appending addrspace(1) global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @foo to ptr), ptr addrspacecast (ptr addrspace(4) @bar to ptr)], section "llvm.metadata"
//
// SPIRV_AMD_CL: @foo = internal addrspace(2) constant i32 42
// SPIRV_AMD_CL: @llvm.used = appending addrspace(1) global [2 x ptr] [ptr addrspacecast (ptr addrspace(2) @foo to ptr), ptr addrspacecast (ptr addrspace(4) @bar to ptr)], section "llvm.metadata"
//
// X86: define internal void @bar() #{{[0-9]}} {
//
// AMDGCN: define internal void @bar() #{{[0-9]}} {
//
// SPIRV: define internal spir_func void @bar() #{{[0-9]}} {
//
// SPIRV_CL: define internal spir_func void @bar() #{{[0-9]}} {
//
// SPIRV_AMD: define internal spir_func void @bar() addrspace(4) #{{[0-9]}} {
//
// SPIRV_AMD_CL: define internal spir_func void @bar() addrspace(4) #{{[0-9]}} {
//
static void __attribute__((__used__)) bar() {
}
