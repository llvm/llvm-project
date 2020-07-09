// Ensure analyzer option 'ctu-import-threshold' is a recognized option.
//
// RUN: %clang_cc1 -analyze -analyzer-config ctu-import-threshold=30 -verify %s
// RUN: %clang_cc1 -analyze -analyzer-config ctu-import-cpp-threshold=30 -verify %s
//
// expected-no-diagnostics
