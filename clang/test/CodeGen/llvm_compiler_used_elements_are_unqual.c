// RUN: %clang_cc1 -triple x86_64-- -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=AMDGCN
// RUN: %clang_cc1 -triple spirv64-- -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=SPIRV
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -x c %s -o - \
// RUN:   | FileCheck %s --check-prefix=SPIRV_AMD

static __attribute__((__used__)) int foo = 42;


// X86: @foo = internal global i32 42, align 4
// X86: @llvm.compiler.used = appending global [2 x ptr] [ptr @foo, ptr @bar], section "llvm.metadata"
//.
// AMDGCN: @foo = internal addrspace(1) global i32 42, align 4
// AMDGCN: @llvm.compiler.used = appending addrspace(1) global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @foo to ptr), ptr @bar], section "llvm.metadata"
//.
// SPIRV: @foo = internal global i32 42, align 4
// SPIRV: @llvm.used = appending addrspace(1) global [2 x ptr] [ptr @foo, ptr @bar], section "llvm.metadata"
//.
// SPIRV_AMD: @foo = internal addrspace(1) global i32 42, align 4
// SPIRV_AMD: @llvm.used = appending addrspace(1) global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @foo to ptr), ptr addrspacecast (ptr addrspace(4) @bar to ptr)], section "llvm.metadata"
//.
// X86-LABEL: define internal void @bar(
// X86-SAME: ) #[[ATTR0:[0-9]+]] {
//
// AMDGCN-LABEL: define internal void @bar(
// AMDGCN-SAME: ) #[[ATTR0:[0-9]+]] {
//
// SPIRV-LABEL: define internal spir_func void @bar(
// SPIRV-SAME: ) #[[ATTR0:[0-9]+]] {
//
// SPIRV_AMD-LABEL: define internal spir_func void @bar(
// SPIRV_AMD-SAME: ) addrspace(4) #[[ATTR0:[0-9]+]] {
//
static void __attribute__((__used__)) bar() {
}
