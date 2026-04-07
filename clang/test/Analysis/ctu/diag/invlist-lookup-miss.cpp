// Test that a valid invocation list without the requested source file silently fails CTU import.
//
// The invocation list has a valid entry for a different path, but not for the
// source file referenced in externalDefMap.txt.
//
// RUN: rm -rf %t && mkdir %t
// RUN: echo '11:c:@F@foo#I# diag-simple.cpp' > %t/externalDefMap.txt
// RUN: echo '"/nonexistent.cpp": ["clang", "/nonexistent.cpp"]' > %t/invocations.yaml
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -analyzer-config ctu-invocation-list=%t/invocations.yaml \
// RUN:   -verify %s

int foo(int);

void test() {
  foo(1); // expected-warning-re{{invocation for '{{.+}}diag-simple.cpp' is missing in the invocation list}}
}
