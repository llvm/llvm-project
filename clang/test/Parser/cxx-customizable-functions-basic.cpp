// RUN: %clang_cc1 -std=c++20 -fcustomizable-functions -fsyntax-only -verify %s

// Basic parsing of 'custom' keyword

// Test simple custom function
custom void test1() { }  // expected-no-diagnostics

// Test with return type
custom int test2() { return 42; }  // expected-no-diagnostics

// Test with parameters
custom void test3(int x) { }  // expected-no-diagnostics

// Test with auto parameter
custom void test4(auto x) { }  // expected-no-diagnostics

// Test in namespace
namespace ns {
    custom void test5() { }  // expected-no-diagnostics
}

// Test without the feature flag should fail
// RUN: not %clang_cc1 -std=c++20 -fno-customizable-functions -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FEATURE
// CHECK-NO-FEATURE: error: unknown type name 'custom'
