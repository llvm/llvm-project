// Test that loading a C++17 AST from a C++14 translation unit produces
// lang_dialect_mismatch. Both are C++ (no lang_mismatch), but CPlusPlus17
// differs, triggering the dialect check, and silently failing CTU import.
//
// RUN: rm -rf %t && mkdir %t
// RUN: %clang_cc1 -std=c++17 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/simple.cpp.ast \
// RUN:   %S/Inputs/simple.cpp
// RUN: echo '11:c:@F@foo#I# simple.cpp.ast' > %t/externalDefMap.txt
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -verify %s

int foo(int);

void test() {
  foo(1); // expected-warning-re{{imported AST from '{{.+}}simple.cpp' had been generated for a different language, current: C++14, imported: C++17}}
}
