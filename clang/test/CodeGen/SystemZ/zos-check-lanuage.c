//
// This is based on Clang :: Frontend/ast-main.c,
// but targeted for z/OS, and checking only that the langauge comes through.
// We do not check that the entire IR is identical.
// We also check C++.

// The %clang_cc1 -emit-pch does the same as %clang -emit-ast.
// The -x ast in the second invocation of each pair confirms we are reading ast.

// RUN: %clang_cc1 -triple s390x-none-zos -emit-pch -o %t.ast %s
// RUN: %clang_cc1 -triple s390x-none-zos -emit-llvm -o - -x ast - < %t.ast | FileCheck --check-prefix=CHECK-C %s

// check that the langage attribute has come through
// CHECK-C: !"zos_cu_language", !"C"}


// RUN: %clang_cc1 -triple s390x-none-zos -emit-pch -o %t.ast -x c++ %s
// RUN: %clang_cc1 -triple s390x-none-zos -emit-llvm -o - -x ast - < %t.ast | FileCheck --check-prefix CHECK-CPP %s

// check that the langage attribute has come through
// CHECK-CPP: !"zos_cu_language", !"C++"}



int main(void) {
  return 0;
}
