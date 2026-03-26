// Test that loading a C++ AST from a C translation unit produces lang_mismatch.
//
// The external definition is compiled as C++ with extern "C" linkage, so the
// USR matches the C function declaration (both use the no-parameter-type USR
// format: c:@F@foo, 8 chars). The lang mismatch leads to a silent CTU import failure.
//
// RUN: rm -rf %t && mkdir %t
// RUN: %clang_cc1 -x c++ -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/simple-extern-c.cpp.ast \
// RUN:   %S/Inputs/simple-extern-c.cpp
// RUN: echo '8:c:@F@foo simple-extern-c.cpp.ast' > %t/externalDefMap.txt
// RUN: %clang_analyze_cc1 -x c -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -verify %s

int foo(int);

void test(void) {
  // expected-no-diagnostics
  foo(1); // no-warning. Ignoring "Language mismatch."
}
