// Verify clang -cc1 accepts a serialized .cir file as input across all of
// the formats CIRGenAction can emit. With LoweringPrepare AST-free the
// .cir path runs the full CIR-to-CIR pipeline, so emit-obj is now valid as
// well.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s \
// RUN:   -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir %t.cir \
// RUN:   -emit-cir -o - | FileCheck %s --check-prefix=ROUNDTRIP
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir %t.cir \
// RUN:   -emit-llvm -o - | FileCheck %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir %t.cir \
// RUN:   -emit-obj -o %t.o
// RUN: llvm-nm %t.o | FileCheck %s --check-prefix=OBJ

int add(int a, int b) { return a + b; }

// ROUNDTRIP: cir.func {{.*}}@add
// ROUNDTRIP: cir.return

// LLVM: define {{.*}} i32 @add(i32 {{.*}}%[[A:.+]], i32 {{.*}}%[[B:.+]])
// LLVM: ret i32

// OBJ: T add
