// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

short b() { return (short){}; }

// CIR-LABEL: b
// CIR: {{%.*}} = cir.alloca !s16i, !cir.ptr<!s16i>, [".compoundliteral"] {alignment = 2 : i64}

// LLVM-LABEL: b
// LLVM: [[RET_P:%.*]] = alloca i16, i64 1, align 2
// LLVM: [[LITERAL:%.*]] =  alloca i16, i64 1, align 2
// LLVM: store i16 0, ptr [[LITERAL]], align 2
// LLVM: [[T0:%.*]] = load i16, ptr [[LITERAL]], align 2
// LLVM: store i16 [[T0]], ptr [[RET_P]], align 2
// LLVM: [[T1:%.*]] = load i16, ptr [[RET_P]], align 2
// LLVM: ret i16 [[T1]]
