// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// C99 inline definition followed by an extern declaration that forces an
// externally visible declaration of the same symbol.

inline int cut_copy(int *dst, int *src) { return *dst = *src; }
extern int cut_copy(int *dst, int *src);

int driver(void) { return 0; }

// CIR: cir.func{{.*}} @cut_copy(
// CIR: cir.return

// LLVM: define{{.*}} @cut_copy(
// LLVM: define{{.*}} @driver(
