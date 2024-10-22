// RUN: %clang_cc1 -x cl -O0 -cl-std=CL2.0 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -x cl -O0 -cl-std=CL2.0 -emit-llvm -triple spir-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPIR
// RUN: %clang_cc1 -x cl -O0 -cl-std=CL2.0 -emit-llvm -triple spir64-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPIR64
// RUN: %clang_cc1 -x cl -O0 -cl-std=CL2.0 -emit-llvm -triple amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefix=AMDGCN
__kernel void testPipe( pipe int test )
{
    int s = sizeof(test);
    // X86: store ptr %test, ptr %test.addr, align 8
    // X86: store i32 8, ptr %s, align 4
    // SPIR: store target("spirv.Pipe", 0) %test, ptr %test.addr, align 4
    // SPIR: store i32 4, ptr %s, align 4
    // SPIR64: store target("spirv.Pipe", 0) %test, ptr %test.addr, align 8
    // SPIR64: store i32 8, ptr %s, align 4
    // AMDGCN: store ptr addrspace(1) %test, ptr %test{{.*}}, align 8
    // AMDGCN: store i32 8, ptr %s{{.*}}, align 4
}
