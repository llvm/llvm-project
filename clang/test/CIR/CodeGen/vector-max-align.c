// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=A64-CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=A64-LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=A64-LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=X64-CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=X64-LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=X64-LLVM --input-file=%t.ll %s

// The vector alignment cap is TargetInfo::getMaxVectorAlign(). AArch64 sets
// it to 128 bits, so a vector wider than 16 bytes is 16-byte aligned.
// x86_64 leaves the cap at 0, so the same vector keeps its natural alignment
// and the module records no #cir.vector_spec entry.

typedef float v4f32 __attribute__((vector_size(16)));
typedef float v8f32 __attribute__((vector_size(32)));
typedef char v17i8 __attribute__((vector_size(17)));

v4f32 g4 = {0};
v8f32 g8 = {0};
v17i8 g17 = {0};

// A64-CIR-DAG: !cir.vector<1 x !cir.int<u, 8>> = #cir.vector_spec<max_align = 128>
// A64-CIR-DAG: cir.global external @g4 = {{.*}}!cir.vector<4 x !cir.float> align(16)
// A64-CIR-DAG: cir.global external @g8 = {{.*}}!cir.vector<8 x !cir.float> align(16)
// A64-CIR-DAG: cir.global external @g17 = {{.*}}!cir.vector<17 x !s8i> align(16)
// A64-LLVM-DAG: @g4 = {{.*}}global <4 x float> zeroinitializer, align 16
// A64-LLVM-DAG: @g8 = {{.*}}global <8 x float> zeroinitializer, align 16
// A64-LLVM-DAG: @g17 = {{.*}}global <17 x i8> zeroinitializer, align 16

// X64-CIR-NOT: #cir.vector_spec
// X64-CIR-DAG: cir.global external @g4 = {{.*}}!cir.vector<4 x !cir.float> align(16)
// X64-CIR-DAG: cir.global external @g8 = {{.*}}!cir.vector<8 x !cir.float> align(32)
// X64-CIR-DAG: cir.global external @g17 = {{.*}}!cir.vector<17 x !s8i> align(32)
// X64-LLVM-DAG: @g4 = {{.*}}global <4 x float> zeroinitializer, align 16
// X64-LLVM-DAG: @g8 = {{.*}}global <8 x float> zeroinitializer, align 32
// X64-LLVM-DAG: @g17 = {{.*}}global <17 x i8> zeroinitializer, align 32
