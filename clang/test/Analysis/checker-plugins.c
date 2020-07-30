// REQUIRES: plugins

// FIXME: This test fails on clang-stage2-cmake-RgSan,
// see also https://reviews.llvm.org/D62445#1613268
// UNSUPPORTED: darwin

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -load %llvmshlibdir/SampleAnalyzerPlugin%pluginext \
// RUN:   -analyzer-checker='example.MainCallChecker'

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

// RUN: %clang_analyze_cc1 %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   -analyzer-checker=debug.ConfigDumper \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-CHECKER-OPTION

// CHECK-CHECKER-OPTION: example.MyChecker:ExampleOption = false

// RUN: %clang_analyze_cc1 %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   -analyzer-checker=debug.ConfigDumper \
// RUN:   -analyzer-config example.MyChecker:ExampleOption=true \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-CHECKER-OPTION-TRUE

// CHECK-CHECKER-OPTION-TRUE: example.MyChecker:ExampleOption = true

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   -analyzer-config example.MyChecker:Example=true \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-NON-EXISTENT-CHECKER-OPTION

// CHECK-NON-EXISTENT-CHECKER-OPTION: (frontend): checker 'example.MyChecker'
// CHECK-NON-EXISTENT-CHECKER-OPTION-SAME: has no option called 'Example'

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config example.MyChecker:Example=true


// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   -analyzer-config example.MyChecker:ExampleOption=example \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-BOOL-VALUE

// CHECK-INVALID-BOOL-VALUE: (frontend): invalid input for checker option
// CHECK-INVALID-BOOL-VALUE-SAME: 'example.MyChecker:ExampleOption', that
// CHECK-INVALID-BOOL-VALUE-SAME: expects a boolean value

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config example.MyChecker:ExampleOption=example

// RUN: %clang_analyze_cc1 %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   -analyzer-checker=debug.ConfigDumper \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config example.MyChecker:ExampleOption=example \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-CORRECTED-BOOL-VALUE

// CHECK-CORRECTED-BOOL-VALUE: example.MyChecker:ExampleOption = false

// RUN: %clang_analyze_cc1 %s \
// RUN:   -load %llvmshlibdir/CheckerOptionHandlingAnalyzerPlugin%pluginext\
// RUN:   -analyzer-checker=example.MyChecker \
// RUN:   -analyzer-checker-option-help \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-CHECKER-OPTION-HELP

// CHECK-CHECKER-OPTION-HELP: example.MyChecker:ExampleOption  (bool) This is an
// CHECK-CHECKER-OPTION-HELP-SAME: example checker opt. (default:
// CHECK-CHECKER-OPTION-HELP-NEXT: false)
