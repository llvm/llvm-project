// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-int-conversions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-int-conversions -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// GNU extensions
typedef void (*FP)(void);
void *f2(void *a, int b) { return a + b; }
// CIR-LABEL: f2
// CIR: %[[PTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: %[[STRIDE:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR: cir.ptr_stride(%[[PTR]] : !cir.ptr<!void>, %[[STRIDE]] : !s32i)

// LLVM-LABEL: f2
// LLVM: %[[PTR:.*]] = load ptr, ptr {{.*}}, align 8
// LLVM: %[[TOEXT:.*]] = load i32, ptr {{.*}}, align 4
// LLVM: %[[STRIDE:.*]] = sext i32 %[[TOEXT]] to i64
// LLVM: getelementptr i8, ptr %[[PTR]], i64 %[[STRIDE]]