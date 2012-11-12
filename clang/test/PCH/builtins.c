// Test this without pch.
// RUN: %clang_cc1 -include %S/builtins.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/builtins.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

// expected-no-diagnostics

void hello() {
  printf("Hello, World!");
}
