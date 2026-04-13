// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o %t.cir 
// RUN: FileCheck --input-file=%t.cir %s

// CHECK:  cir.module_asm = [".globl bar", ".globl foo"]
__asm (".globl bar");
__asm (".globl foo");
