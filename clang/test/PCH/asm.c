// Test this without pch.
// RUN: %clang_cc1 -triple i386-unknown-unknown -include %S/asm.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-pch -o %t %S/asm.h
// RUN: %clang_cc1 -triple i386-unknown-unknown -include-pch %t -fsyntax-only -verify %s 

// expected-no-diagnostics

void call_f(void) { f(); }

void call_clobbers(void) { clobbers(); }
