// RUN: %clang_cc1 -x cl -O0 -cl-std=CL2.0 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -x cl -O0 -cl-std=CL2.0 -emit-llvm -triple spir-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPIR
// RUN: %clang_cc1 -x cl -O0 -cl-std=CL2.0 -emit-llvm -triple spir64-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPIR64
// RUN: %clang_cc1 -x cl -O0 -cl-std=CL2.0 -emit-llvm -triple amdgcn-amd-amdhsa %s -o - | FileCheck %s --check-prefix=AMDGCN
__kernel void testPipe( pipe int test )
{
    int s = sizeof(test);
    // X86: store %opencl.pipe_ro_t* %test, %opencl.pipe_ro_t** %test.addr, align 8
    // X86: store i32 8, i32* %s, align 4
    // SPIR: store %opencl.pipe_ro_t addrspace(1)* %test, %opencl.pipe_ro_t addrspace(1)** %test.addr, align 4
    // SPIR: store i32 4, i32* %s, align 4
    // SPIR64: store %opencl.pipe_ro_t addrspace(1)* %test, %opencl.pipe_ro_t addrspace(1)** %test.addr, align 8
    // SPIR64: store i32 8, i32* %s, align 4
    // AMDGCN: store %opencl.pipe_ro_t addrspace(1)* %test, %opencl.pipe_ro_t addrspace(1)* addrspace(5)* %test.addr, align 8
    // AMDGCN: store i32 8, i32 addrspace(5)* %s, align 4
}
