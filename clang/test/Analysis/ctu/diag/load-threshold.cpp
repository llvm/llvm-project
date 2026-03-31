// Test that exceeding ctu-import-cpp-threshold produces load_threshold_reached,
// which is silently fails AST import.
//
// With threshold=1, the first external AST (foo) is loaded successfully.
// The second lookup (bar) finds the threshold exhausted and reports the error.
//
// RUN: rm -rf %t && mkdir %t
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/simple.cpp.ast \
// RUN:   %S/Inputs/simple.cpp
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/bar.cpp.ast \
// RUN:   %S/Inputs/bar.cpp
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/third.cpp.ast \
// RUN:   %S/Inputs/third.cpp
// RUN: echo '11:c:@F@foo#I# simple.cpp.ast' > %t/externalDefMap.txt
// RUN: echo '11:c:@F@bar#I# bar.cpp.ast' >> %t/externalDefMap.txt
// RUN: echo '13:c:@F@third#I# third.cpp.ast' >> %t/externalDefMap.txt
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t \
// RUN:   -analyzer-config ctu-import-cpp-threshold=1 \
// RUN:   -Rctu \
// RUN:   -verify %s

// foo is loaded successfully (first load, within threshold).
// bar hits the threshold (no telemetry emitted for it).

int foo(int);
int bar(int);
int third(int);

void test() {
  foo(1);
  bar(1); // expected-remark {{reached a the CTU-import threshold before trying to import definition}}
  bar(2); // expected-remark {{reached a the CTU-import threshold before trying to import definition}}
  third(1); // expected-remark {{reached a the CTU-import threshold before trying to import definition}}
}
