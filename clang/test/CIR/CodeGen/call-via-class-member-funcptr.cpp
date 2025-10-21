// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

class A {
public:
  static char *b(int);
};

int h=0;

class F {
public:
  const char *b();
  A g;
};

const char *F::b() { return g.b(h); }

void fn1() { F f1; }

// CIR: cir.func {{.*}} @_ZN1F1bEv
// CIR:   %[[H_PTR:.*]] = cir.get_global @h : !cir.ptr<!s32i>
// CIR:   %[[H_VAL:.*]] = cir.load{{.*}} %[[H_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[RET:.*]] = cir.call @_ZN1A1bEi(%[[H_VAL]]) : (!s32i) -> !cir.ptr<!s8i>

// LLVM: define {{.*}} ptr @_ZN1F1bEv
// LLVM:   %[[VAR_H:.*]] = load i32, ptr @h
// LLVM:   %[[RET:.*]] = call ptr @_ZN1A1bEi(i32 %[[VAR_H]])

// OGCG: define {{.*}} ptr @_ZN1F1bEv
// OGCG:   %[[VAR_H:.*]] = load i32, ptr @h
// OGCG:   %[[RET:.*]] = call noundef ptr @_ZN1A1bEi(i32 noundef %[[VAR_H]])
