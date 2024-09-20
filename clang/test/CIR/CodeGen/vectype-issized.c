// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir -emit-cir -target-feature +neon %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android24  -fclangir -emit-llvm -target-feature +neon %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef __attribute__((neon_vector_type(8))) short  c;
void d() { c a[8]; }

// CIR-LABEL: d
// CIR: {{%.*}} = cir.alloca !cir.array<!cir.vector<!s16i x 8> x 8>,
// CIR-SAME: !cir.ptr<!cir.array<!cir.vector<!s16i x 8> x 8>>, ["a"]
// CIR-SAME: {alignment = 16 : i64}

// LLVM-LABEL: d
// LLVM: {{%.*}} = alloca [8 x <8 x i16>], i64 1, align 16
