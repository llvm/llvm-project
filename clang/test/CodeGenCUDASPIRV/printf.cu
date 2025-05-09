// RUN: %clang_cc1 -fcuda-is-device -triple spirv32 -o - -emit-llvm -x cuda %s  | FileCheck --check-prefix=CHECK-SPIRV32 %s
// RUN: %clang_cc1 -fcuda-is-device -triple spirv64 -o - -emit-llvm -x cuda %s  | FileCheck --check-prefix=CHECK-SPIRV64 %s

// CHECK-SPIRV32: @.str = private unnamed_addr addrspace(4) constant [13 x i8] c"Hello World\0A\00", align 1
// CHECK-SPIRV64: @.str = private unnamed_addr addrspace(1) constant [13 x i8] c"Hello World\0A\00", align 1

extern "C" __attribute__((device)) int printf(const char* format, ...);

__attribute__((global)) void printf_kernel() {
  printf("Hello World\n");
}
