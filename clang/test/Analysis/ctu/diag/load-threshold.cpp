// Test that exceeding ctu-import-cpp-threshold produces load_threshold_reached,
// which silently fails AST import.
//
// With threshold=1, the first external AST (foo) is loaded successfully.
// The second lookup (bar) finds the threshold exhausted and emits a remark once.
// All subsequent threshold-blocked lookups fail silently, including those in a
// second analysis entry point (test2): the remark is not repeated, and the
// already-cached AST for foo remains accessible.
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
// bar is the first to hit the threshold; the remark is emitted once.
// Subsequent threshold-blocked lookups (bar(2), third) fail silently.
//
// test2() is a second entry point analyzed after test(). It is defined first
// to be analyzed last.

int foo(int);
int bar(int);
int third(int);

// In a second entry point the threshold state persists from test(): the remark
// is not repeated, foo's AST is still accessible from cache, and new
// threshold-blocked lookups (bar, third) fail silently.
void test2() {
  foo(1);    // no remark: foo's AST was cached during test()'s analysis
  bar(1);    // no remark: threshold already reported in test()
  third(1);  // no remark: threshold already reported in test()
}

void test() {
  foo(1);
  bar(1); // expected-remark {{reached a the CTU-import threshold before trying to import definition}}
  bar(2); // no remark: threshold already reported
  third(1); // no remark: threshold already reported
}
