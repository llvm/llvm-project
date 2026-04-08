// Test that missing externalDefMap.txt produces err_ctu_error_opening at call site.
//
// RUN: rm -rf %t && mkdir %t
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -analyzer-config ctu-index-name=non-existing.txt \
// RUN:   -verify %s

int foo(int);

void test() {
  foo(1); // expected-error-re{{error opening '{{.+}}non-existing.txt': required by the CrossTU functionality}}
}
