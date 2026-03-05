// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -emit-mlir %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang -target x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir %s -o - | FileCheck %s -check-prefix=CORE

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-mlir=llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-mlir=core %s -o - 2>&1 | FileCheck %s -check-prefix=CORE_ERR

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir=llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir=core %s -o - | FileCheck %s -check-prefix=CORE

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-mlir=cir %s -o - | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-mlir=cir-flat %s -o - | FileCheck %s -check-prefix=CIR_FLAT

// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -emit-mlir %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang -target x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir %s -o - | FileCheck %s -check-prefix=CORE

// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -emit-mlir %s -o - -### 2>&1 | FileCheck %s -check-prefix=OPTS_LLVM
// RUN: %clang -target x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir %s -o - -### 2>&1 | FileCheck %s -check-prefix=OPTS_CORE

// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -emit-mlir=llvm %s -o - -###  2>&1 | FileCheck %s -check-prefix=OPTS_LLVM
// RUN: %clang -target x86_64-unknown-linux-gnu -fno-clangir-direct-lowering -emit-mlir=core %s -o - -### 2>&1 | FileCheck %s -check-prefix=OPTS_CORE
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -emit-mlir=cir %s -o - -###  2>&1 | FileCheck %s -check-prefix=OPTS_CIR
// RUN: %clang -target x86_64-unknown-linux-gnu -fclangir -emit-mlir=cir-flat %s -o - -###  2>&1 | FileCheck %s -check-prefix=OPTS_CIR_FLAT

int foo(int a, int b) {
    int c;
    if (a) {
      c = a;
    }
    c = b;
    return c;
}

// LLVM: llvm.func @foo
// CORE: func.func @foo
// CIR: cir.func {{.*}} @foo
// CIR: cir.scope
// CIR_FLAT: cir.func {{.*}} @foo
// CIR_FLAT: ^bb1
// CIR_FLAT-NOT: cir.scope
// CORE_ERR: ClangIR direct lowering is incompatible with emitting of MLIR standard dialects
// OPTS_LLVM: "-emit-mlir=llvm"
// OPTS_CORE: "-emit-mlir=core"
// OPTS_CIR: "-emit-mlir=cir"
// OPTS_CIR_FLAT: "-emit-mlir=cir-flat"