// RUN: %clang_cc1 %s -fexperimental-overflow-behavior-types -Wimplicit-overflow-behavior-conversion-assignment-pedantic -verify -fsyntax-only -std=c11

// Test that -Wimplicit-overflow-behavior-conversion-assignment-pedantic only warns
// for unsigned wrap to unsigned conversions during assignment

void test_assignment_pedantic() {
  unsigned int __ob_wrap uwrap = 42;
  unsigned int ureg = uwrap; // expected-warning {{implicit conversion from '__ob_wrap unsigned int' to 'unsigned int' during assignment discards overflow behavior}}

  int __ob_wrap swrap = 42;
  int sreg = swrap;

  unsigned int __ob_trap utrap = 42;
  unsigned int ureg2 = utrap;

  unsigned int ureg3 = (unsigned int)uwrap;
}
