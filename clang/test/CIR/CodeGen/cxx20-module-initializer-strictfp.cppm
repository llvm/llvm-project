// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -ffp-exception-behavior=strict -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -ffp-exception-behavior=strict -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM

// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir %s -o - | FileCheck %s --check-prefix=CIR-DEFAULT
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-DEFAULT
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM-DEFAULT

// A named-module interface unit gets an initializer function even with
// nothing to run. It is strictfp when the TU-wide default FP environment is
// constrained.

export module A;

// CIR: cir.func private @_ZGIW1A() attributes {{{.*}}strictfp}

// LLVM: define void @_ZGIW1A() #[[INIT:[0-9]+]]
// LLVM: attributes #[[INIT]] = { {{.*}}strictfp

// The NOT patterns name the attribute lists, since the path of this file can
// appear in the output.
// CIR-DEFAULT: cir.func private @_ZGIW1A()
// CIR-DEFAULT-NOT: attributes {{{.*}}strictfp

// LLVM-DEFAULT: define void @_ZGIW1A()
// LLVM-DEFAULT-NOT: attributes #{{[0-9]+}} = {{.*}}strictfp
