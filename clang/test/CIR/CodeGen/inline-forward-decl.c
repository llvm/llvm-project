// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=gnu17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=gnu17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=gnu17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

inline int helper(int);
void user(void) {
  helper(1);
}
inline int helper(int x) {
  return x;
}

__attribute__((weak)) void weak_fn(void) {}

// CIR-DAG: cir.func{{.*}}@helper
// CIR-DAG: cir.func{{.*}}@user
// CIR-DAG: cir.func{{.*}}weak @weak_fn
// CIR-DAG: cir.call @helper

// LLVM: define{{.*}}@user
// LLVM: call{{.*}}@helper
// LLVM: define weak void @weak_fn()
