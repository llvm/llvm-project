// Test that an empty invocation list file silently fails CTU import.
//
// Note: invocation_list_empty (index_error_code::invocation_list_empty) is
// dead code: llvm::yaml::Stream::begin() always creates at least one Document,
// so FirstInvocationFile == InvocationFile.end() is never true. An empty file
// reaches the !DocumentRoot branch instead, producing invocation_list_wrong_format.
//
// RUN: rm -rf %t && mkdir %t
// RUN: echo '11:c:@F@foo#I# simple.cpp' > %t/externalDefMap.txt
// RUN: touch %t/invocations.yaml
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
