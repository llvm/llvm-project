// RUN: %clang_cc1 %s -verify -Wno-conversion -Wno-c++11-narrowing -Wimplicit-int-float-conversion

void testNoWarningOnNarrowing() {
  // Test that we do not issue duplicated warnings for
  // C++11 narrowing.
  float a = {222222222222L}; // expected-no-diagnostics

  long b = 222222222222L;
  float c = {b}; // expected-no-diagnostics
}
