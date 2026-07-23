// RUN: %clang_cc1 -triple x86_64-apple-darwin -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

extern void weak_import_fn(void) __attribute__((weak_import));
void user(void) {
  weak_import_fn();
}

// CIR-DAG: cir.func{{.*}}extern_weak{{.*}}@weak_import_fn
// CIR-DAG: cir.func{{.*}}@user
// CIR: cir.call @weak_import_fn

// LLVM-DAG: declare extern_weak void @weak_import_fn()
// LLVM-DAG: define{{.*}}@user
// LLVM-DAG: call{{.*}}@weak_import_fn
