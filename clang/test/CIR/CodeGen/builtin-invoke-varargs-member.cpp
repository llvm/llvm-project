// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=OGCG --input-file=%t.ll

struct S {
  int foo(int, ...) { return 42; }
};

void test(S &s) {
  int (S::*p)(int, ...) = nullptr;
  (void)__builtin_invoke(p, s, 1);
}

// CIR: cir.func{{.*}}@_Z4testR1S
// CIR-DAG: !cir.func<(!cir.ptr<!void>, !s32i, ...) -> !s32i>
// CIR: cir.call %{{.*}}(%{{.*}}, %{{.*}}) : (!cir.ptr<!cir.func<(!cir.ptr<!void>, !s32i, ...) -> !s32i>>, !cir.ptr<!void> {{.*}}, !s32i {{.*}}) -> (!s32i

// LLVM: call noundef i32 (ptr, i32, ...) %{{.*}}(ptr noundef %{{.*}}, i32 noundef 1)

// OGCG: call noundef i32 (ptr, i32, ...) %{{.*}}(ptr noundef{{.*}} %{{.*}}, i32 noundef 1)
