// RUN: rm -f %t.pch
// RUN: %clang_cc1 -emit-pch -o %t.pch %S/Inputs/chain-decls1.h
// RUN: %clang_cc1 -include-pch %t.pch -fsyntax-only -MT %s.o -dependency-file - %s | FileCheck %s
// CHECK: chain-decls1.h

int main(void) {
  f();
  return 0;
}
