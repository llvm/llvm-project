// Test that a missing invocation list silently fails CTU import.
//
// The externalDefMap.txt maps foo to a non-.ast source path, triggering
// loadFromSource(). The invocation list path points to a nonexistent file.
//
// RUN: rm -rf %t && mkdir %t
// RUN: echo '11:c:@F@foo#I# simple.cpp' > %t/externalDefMap.txt
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -analyzer-config ctu-invocation-list=%t/nonexistent.yaml \
// RUN:   -verify %s

int foo(int);

void test() {
  // expected-no-diagnostics
  foo(1); // no-warning. Ignoring "Invocation list file is not found."
}
