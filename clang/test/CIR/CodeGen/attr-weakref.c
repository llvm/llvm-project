// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// A weakref declaration that is never referenced should produce no output for
// either the alias or its target. Only the unrelated definition below should
// be emitted.
void weakref_target(void);
static void weakref_alias(void) __attribute__((weakref("weakref_target")));

void unrelated(void) {}

// The weakref alias produces no output by itself.
// CIR-NOT: @weakref_alias
// CIR-NOT: @weakref_target
// LLVM-NOT: @weakref_alias
// LLVM-NOT: @weakref_target
// OGCG-NOT: @weakref_alias
// OGCG-NOT: @weakref_target

// CIR: cir.func{{.*}}@unrelated
// LLVM: define{{.*}}@unrelated
// OGCG: define{{.*}}@unrelated
