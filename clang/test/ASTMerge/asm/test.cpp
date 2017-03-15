// RUN: %clang_cc1 -triple i386-unknown-unknown -fcxx-exceptions -emit-pch -o %t.1.ast %S/Inputs/asm-function.cpp
// RUN: %clang_cc1 -triple i386-unknown-unknown -fcxx-exceptions -ast-merge %t.1.ast -fsyntax-only -verify %s
// expected-no-diagnostics

void testAsmImport() {
  asmFunc(12, 42);
  asmFunc2(42);
}
