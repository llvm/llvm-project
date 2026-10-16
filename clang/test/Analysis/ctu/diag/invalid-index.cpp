// Test that a malformed externalDefMap.txt produces err_extdefmap_parsing error.
//
// RUN: rm -rf %t && mkdir %t
// RUN: echo 'this is invalid' > %t/externalDefMap.txt
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -verify %s

int foo(int);

void test() {
  foo(1); // expected-error-re{{error parsing index file: '{{.+}}externalDefMap.txt' line: 1 '<USR-Length>:<USR> <File-Path>' format expected}}
}
