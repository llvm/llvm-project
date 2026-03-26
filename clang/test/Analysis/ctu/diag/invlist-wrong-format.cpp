// Test that a malformed invocation list produces silently fails CTU import.
//
// A YAML sequence (- item) at the root is not a mapping, so dyn_cast to
// MappingNode fails, triggering invocation_list_wrong_format.
//
// RUN: rm -rf %t && mkdir %t
// RUN: echo '11:c:@F@foo#I# simple.cpp' > %t/externalDefMap.txt
// RUN: echo '- just_a_list_item' > %t/invocations.yaml
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -analyzer-config ctu-invocation-list=%t/invocations.yaml \
// RUN:   -verify %s

int foo(int);

void test() {
  // expected-no-diagnostics
  foo(1); // no-warning. Ignoring "Invocation list file is in wrong format."
}
