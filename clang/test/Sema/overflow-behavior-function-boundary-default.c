// RUN: %clang_cc1 %s -fexperimental-overflow-behavior-types -fsyntax-only -verify

// Test that function boundary warnings are off by default
// expected-no-diagnostics

void func_no_obt(int x) {}
void func_no_obt_unsigned(unsigned int x) {}

void test_function_boundary_default() {
  int __ob_wrap w = 42;
  unsigned int __ob_wrap uw = 42;

  // These should NOT warn by default
  func_no_obt(w);
  func_no_obt_unsigned(uw);
}
