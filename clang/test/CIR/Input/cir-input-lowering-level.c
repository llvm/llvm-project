// The module records how far it has been through the CIR-to-CIR lowering
// passes, so that CIR read back is not lowered twice or out of order.

// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=FULL
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-cir %s -o - | FileCheck %s --check-prefix=NOCC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -clangir-disable-passes -emit-cir %s -o %t.early.cir
// RUN: FileCheck %s --check-prefix=EARLY --input-file=%t.early.cir

// FULL:  cir.lowering_level = #cir.lowering_level<callconv_lowered>
// NOCC:  cir.lowering_level = #cir.lowering_level<prepared>
// EARLY: cir.lowering_level = #cir.lowering_level<none>

// CIR that has not been through the CIR-to-CIR passes cannot be lowered.
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t.early.cir -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOTLOWERED
// NOTLOWERED: error: 'cir-flat-to-llvm' expects a module at lowering level 'prepared' or later, but the module is at 'none'

void f(void) {}
