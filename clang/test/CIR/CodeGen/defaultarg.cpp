// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -std=c++17 %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -std=c++17 %s -o %t-cir.ll
// RUN: FileCheck %s --input-file=%t-cir.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -std=c++17 %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=OGCG

void bar(const int &i = 42);

void foo() {
  bar();
}

// CIR: cir.func {{.*}} @_Z3foov()
// CIR:   cir.scope {
// CIR:     %[[TMP0:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp0"]
// CIR:     %[[TMP1:.*]] = cir.const #cir.int<42>
// CIR:     cir.store{{.*}} %[[TMP1]], %[[TMP0]]
// CIR:     cir.call @_Z3barRKi(%[[TMP0]])
// CIR:   }

// LLVM: define{{.*}} @_Z3foov()
// LLVM:   %[[TMP0:.*]] = alloca i32
// LLVM:   br label %[[SCOPE_LABEL:.*]]
// LLVM: [[SCOPE_LABEL]]:
// LLVM:   store i32 42, ptr %[[TMP0]]
// LLVM:   call void @_Z3barRKi(ptr %[[TMP0]])

// OGCG: define{{.*}} @_Z3foov()
// OGCG:   %[[TMP0:.*]] = alloca i32
// OGCG:   store i32 42, ptr %[[TMP0]]
// OGCG:   call void @_Z3barRKi(ptr {{.*}} %[[TMP0]])
