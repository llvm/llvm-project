// Test that loading of invalid AST dump leads CTU import failure and a note on stderr.
// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: touch %t/invalid-ast-other.cpp.ast

// RUN: cp %S/Inputs/invalid-ast-other.cpp.externalDefMap.ast-dump.txt %t/externalDefMap.txt

// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -std=c++17 \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -verify %s 2>&1 | FileCheck %s

// CHECK: error: unable to load precompiled file

// FIXME: this is misleading
// CHECK: CTU loaded AST file: invalid-ast-other.cpp.ast

void external();

void trigger() {
  // expected-no-diagnostics
  external(); // no-warning
}
