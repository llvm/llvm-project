// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR-CL30
// RUN: %clang_cc1 -cl-std=CL3.0 -O0 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM-CL30
// RUN: %clang_cc1 -cl-std=CL1.2 -O0 -fclangir -emit-cir -triple spirv64-unknown-unknown %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR-CL12
// RUN: %clang_cc1 -cl-std=CL1.2 -O0 -fclangir -emit-llvm -triple spirv64-unknown-unknown %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM-CL12

// CIR-CL30: module {{.*}} attributes {{{.*}}cir.cl.version = #cir.cl.version<3, 0>
// LLVM-CL30: !opencl.ocl.version = !{![[MDCL30:[0-9]+]]}
// LLVM-CL30: ![[MDCL30]] = !{i32 3, i32 0}

// CIR-CL12: module {{.*}} attributes {{{.*}}cir.cl.version = #cir.cl.version<1, 2>
// LLVM-CL12: !opencl.ocl.version = !{![[MDCL12:[0-9]+]]}
// LLVM-CL12: ![[MDCL12]] = !{i32 1, i32 2}
