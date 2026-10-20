// Test that a malformed invocation list entry on a non-first line reports the
// correct line number. The first mapping entry is valid; the second has a
// scalar value instead of a sequence, triggering invocation_list_wrong_format.
//
// RUN: rm -rf %t && mkdir %t
// RUN: echo '11:c:@F@foo#I# simple.cpp' > %t/externalDefMap.txt
// RUN: printf '/tmp/valid.cpp:\n  - clang++\n/tmp/bad.cpp: not_a_sequence\n' > %t/invocations.yaml
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -analyzer-config ctu-invocation-list=%t/invocations.yaml \
// RUN:   -verify %s

int foo(int);

void test() {
  foo(1); // expected-error-re{{error parsing invocation list file: '{{.+}}invocations.yaml' line: 3 '<source-file>: [<compiler>, <arg1>, ...]' YAML mapping format expected}}
}
