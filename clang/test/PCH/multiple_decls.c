// Test this without pch.
// RUN: %clang_cc1 -include %S/multiple_decls.h -ast-print -o - %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/multiple_decls.h
// RUN: %clang_cc1 -include-pch %t -ast-print -o - %s 

void f0(char c) {
  wide(c);
}

struct wide w;
struct narrow n;

void f1(int i) {
  narrow(i);
}
