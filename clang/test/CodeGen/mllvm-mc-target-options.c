// REQUIRES: x86-registered-target
///
/// Clang registers LLVM's machine-code target options (via
/// RegisterMCTargetOptionsFlags) so they can be set through -mllvm. An MC option
/// that has no dedicated clang flag takes effect directly; an MC option that also
/// has a clang flag is governed by the clang flag, which takes precedence over
/// -mllvm.

/// -asm-show-inst has no clang flag, so -mllvm controls it and the emitted
/// assembly gains <MCInst ...> comments.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S %s -o - \
// RUN:   | FileCheck %s --check-prefix=NO-SHOW-INST
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -S -mllvm -asm-show-inst %s -o - \
// RUN:   | FileCheck %s --check-prefix=SHOW-INST
// NO-SHOW-INST-NOT: <MCInst
// SHOW-INST: <MCInst

/// -x86-relax-relocations also has a clang flag (-mrelax-relocations), which wins
/// over -mllvm: with -mrelax-relocations=no the GOT load keeps its non-relaxable
/// relocation even though -mllvm -x86-relax-relocations requests relaxation.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -mrelocation-model pic \
// RUN:   -mrelax-relocations=no -mllvm -x86-relax-relocations -emit-obj %s -o %t
// RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=PRECEDENCE
// PRECEDENCE:     R_X86_64_GOTPCREL foo
// PRECEDENCE-NOT: R_X86_64_REX_GOTPCRELX foo

extern int foo;
int *f(void) { return &foo; }
