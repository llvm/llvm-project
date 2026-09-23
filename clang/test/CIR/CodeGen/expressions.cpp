// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

void test(int a) {
  // Should generate LValue parenthesis expression.
  (a) = 1;
}

// CIR: cir.func {{.*}} @{{.+}}test
// CIR: %[[CONST:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[CONST]], %{{.+}} : !s32i, !cir.ptr<!s32i>

// LLVM: define dso_local void @_Z4testi(i32 noundef %0)
// LLVM:   store i32 1, ptr %{{.+}}, align 4
// LLVM:   ret void

// OGCG: define dso_local void @_Z4testi(i32 noundef %a)
// OGCG:   store i32 1, ptr %{{.+}}, align 4
// OGCG:   ret void
