// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -fembed-bitcode=marker -emit-llvm %s -o - \
// RUN:   | FileCheck %s

// CHECK: @llvm.embedded.module = private addrspace(1) constant [0 x i8] zeroinitializer, section ".llvmbc", align 1
// CHECK: @llvm.cmdline = private addrspace(1) constant [{{[0-9]+}} x i8] c"{{.*}}", section ".llvmcmd", align 1
// CHECK: @llvm.compiler.used = {{.*}}@llvm.embedded.module{{.*}}@llvm.cmdline

__attribute__((global)) void k() {}
