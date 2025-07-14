// Ensure analyzer option 'ctu-import-threshold' is a recognized option.
//
// RUN: %clang_analyze_cc1 -analyzer-config ctu-import-threshold=30 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-config ctu-import-cpp-threshold=30 -verify %s
//
// expected-no-diagnostics
