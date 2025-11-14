// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test basic usage - valid
__attribute__((flatten_depth(3)))
void test_valid() {
}

// Test attribute on non-function - should error
__attribute__((flatten_depth(3))) int x; // expected-error {{'flatten_depth' attribute only applies to functions}}

// Test depth = 0 - should error (depth must be >= 1)
__attribute__((flatten_depth(0))) // expected-error {{'flatten_depth' attribute must be greater than 0}}
void test_depth_zero() {
}
