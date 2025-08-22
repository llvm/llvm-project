// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void f1(void) {
  _Atomic(int) x = 42;
}

// CIR-LABEL: @f1
// CIR:         %[[SLOT:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR-NEXT:    %[[INIT:.+]] = cir.const #cir.int<42> : !s32i
// CIR-NEXT:    cir.store align(4) %[[INIT]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR:       }

// LLVM-LABEL: @f1
// LLVM:         %[[SLOT:.+]] = alloca i32, i64 1, align 4
// LLVM-NEXT:    store i32 42, ptr %[[SLOT]], align 4
// LLVM:       }

// OGCG-LABEL: @f1
// OGCG:         %[[SLOT:.+]] = alloca i32, align 4
// OGCG-NEXT:    store i32 42, ptr %[[SLOT]], align 4
// OGCG:       }

void f2(void) {
  _Atomic(int) x;
  __c11_atomic_init(&x, 42);
}

// CIR-LABEL: @f2
// CIR:         %[[SLOT:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x"] {alignment = 4 : i64}
// CIR-NEXT:    %[[INIT:.+]] = cir.const #cir.int<42> : !s32i
// CIR-NEXT:    cir.store align(4) %[[INIT]], %[[SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR:       }

// LLVM-LABEL: @f2
// LLVM:         %[[SLOT:.+]] = alloca i32, i64 1, align 4
// LLVM-NEXT:    store i32 42, ptr %[[SLOT]], align 4
// LLVM:       }

// OGCG-LABEL: @f2
// OGCG:         %[[SLOT:.+]] = alloca i32, align 4
// OGCG-NEXT:    store i32 42, ptr %[[SLOT]], align 4
// OGCG:       }
