// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-assume-sane-operator-new -fclangir -emit-llvm %s -o %t-nosane-cir.ll
// RUN: FileCheck --check-prefix=NOSANE --input-file=%t-nosane-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-assume-sane-operator-new -emit-llvm %s -o %t-nosane.ll
// RUN: FileCheck --check-prefix=NOSANE --input-file=%t-nosane.ll %s

void *test_new() { return new int; }

// CIR: cir.call @_Znwm(%{{.*}}) side_effect(inaccessible_or_errno) {{{.*}}builtin} : {{.*}} -> (!cir.ptr<!void> {llvm.noalias{{.*}}})

// LLVM: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4) [[ATTR:#[0-9]+]]
// OGCG: call noalias noundef nonnull ptr @_Znwm(i64 noundef 4) [[ATTR:#[0-9]+]]
// LLVM: attributes [[ATTR]] = { builtin allocsize(0) memory(inaccessiblemem: readwrite, errnomem: write) }
// OGCG: attributes [[ATTR]] = { builtin allocsize(0) memory(inaccessiblemem: readwrite, errnomem: write) }

// NOSANE: call noundef nonnull ptr @_Znwm(i64 noundef 4) [[NOSANE_ATTR:#[0-9]+]]
// NOSANE-NOT: call noalias
// NOSANE: attributes [[NOSANE_ATTR]] = { builtin allocsize(0) }
// NOSANE-NOT: inaccessiblemem
