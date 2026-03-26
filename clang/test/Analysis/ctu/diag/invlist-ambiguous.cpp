// Test that duplicate keys in the invocation list silently fails CTU import.
//
// RUN: rm -rf %t && mkdir %t
// RUN: echo '11:c:@F@foo#I# simple.cpp' > %t/externalDefMap.txt
// RUN: echo '"/some/path.cpp": ["clang", "/some/path.cpp"]' > %t/invocations.yaml
// RUN: echo '"/some/path.cpp": ["clang", "/some/path.cpp"]' >> %t/invocations.yaml
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -analyzer-config ctu-invocation-list=%t/invocations.yaml \
// RUN:   -verify %s

int foo(int);

void test() {
  // expected-no-diagnostics
  foo(1); // no-warning. Ignoring "Invocation list file contains multiple references to the same source file."
}
