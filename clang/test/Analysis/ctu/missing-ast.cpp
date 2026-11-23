// Test that when the referenced AST file is missing, CTU import fails with a note in stderr.
// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: cp %S/Inputs/missing-ast.cpp.externalDefMap.ast-dump.txt %t/externalDefMap.txt

// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -std=c++17 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -verify %s 2>&1 | FileCheck %s

// CHECK: error: unable to load precompiled file

// FIXME: this is misleading
// CHECK: CTU loaded AST file: wrong-missing-ast.cpp.ast

void external();

void trigger() {
  external(); // expected-error{{import of an external symbol for CTU failed: Failed to load external AST source.}}
}
