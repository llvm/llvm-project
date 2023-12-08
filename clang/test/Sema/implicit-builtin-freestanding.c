// RUN: %clang_cc1 -fsyntax-only -verify -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify -fno-builtin %s
// RUN: %clang_cc1 -fsyntax-only -verify -fno-builtin-malloc %s
// expected-no-diagnostics

int malloc(int a) { return a; }

