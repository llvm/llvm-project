// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -emit-llvm -fcuda-is-device -fembed-bitcode=marker -x hip %s -o - \
// RUN:   | FileCheck %s --check-prefix=CHECK

// CHECK: @llvm.embedded.module = private addrspace(1) constant [0 x i8] zeroinitializer, section ".llvmbc", align 1
// CHECK-NEXT: @llvm.cmdline = private addrspace(1) constant [{{[0-9]+}} x i8] c"{{.*}}", section ".llvmcmd", align 1
// CHECK-NEXT: @llvm.compiler.used = appending addrspace(1) global [5 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @foo.managed to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @foo to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__hip_cuid_ to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @llvm.embedded.module to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @llvm.cmdline to ptr addrspace(4))], section "llvm.metadata"

__attribute__((managed)) int foo = 42;
