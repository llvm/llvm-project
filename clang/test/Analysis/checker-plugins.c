// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -load %llvmshlibdir/SampleAnalyzerPlugin%pluginext \
// RUN:   -analyzer-checker='example.MainCallChecker'

// REQUIRES: plugins

// Test that the MainCallChecker example analyzer plugin loads and runs.

int main();

void caller() {
  main(); // expected-warning {{call to main}}
}

// RUN: %clang_analyze_cc1 %s \
// RUN:   -load %llvmshlibdir/CheckerDependencyHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.DependendentChecker \
// RUN:   -analyzer-list-enabled-checkers \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-IMPLICITLY-ENABLED

// CHECK-IMPLICITLY-ENABLED: example.Dependency
// CHECK-IMPLICITLY-ENABLED: example.DependendentChecker

// RUN: %clang_analyze_cc1 %s \
// RUN:   -load %llvmshlibdir/CheckerDependencyHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.DependendentChecker \
// RUN:   -analyzer-disable-checker=example.Dependency \
// RUN:   -analyzer-list-enabled-checkers \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-IMPLICITLY-DISABLED

// CHECK-IMPLICITLY-DISABLED-NOT: example.Dependency
// CHECK-IMPLICITLY-DISABLED-NOT: example.DependendentChecker

// RUN: %clang_analyze_cc1 %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-CHECKER-OPTION-OUTPUT

// CHECK-CHECKER-OPTION-OUTPUT: Example option is set to false

// RUN: %clang_analyze_cc1 %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   -analyzer-config example.MyChecker:ExampleOption=true \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-CHECKER-OPTION-OUTPUT-TRUE

// CHECK-CHECKER-OPTION-OUTPUT-TRUE: Example option is set to true
