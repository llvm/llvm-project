// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

template <typename T>
concept C = sizeof(T) % 2 == 0;

void concept_specialization() { bool a = C<int>; }

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["a", init]
// CIR: %[[CONST_TRUE:.*]] = cir.const #true
// CIR: cir.store {{.*}} %[[CONST_TRUE]], %[[A_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: store i8 1, ptr %[[A_ADDR]], align 1

// OGCG: %[[A_ADDR:.*]] = alloca i8, align 1
// OGCG: store i8 1, ptr %[[A_ADDR]], align 1
