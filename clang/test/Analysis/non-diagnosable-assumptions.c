// RUN: %clang_analyze_cc1 -w -analyzer-checker=core.DivideZero -analyzer-output=text -verify %s

// This test file verifies the "Assuming..." diagnostic pieces that are being
// reported when the branch condition was too complicated to explain.
// Therefore, if your change replaces the generic "Assuming the condition is
// true" with a more specific message, causing this test to fail, the condition
// should be replaced with a more complicated condition that we still cannot
// properly explain to the user. Once we reach the point at which all conditions
// are "diagnosable", this test (or this note) should probably be removed,
// together with the code section that handles generic messages for
// non-diagnosable conditions.

// Function calls are currently non-diagnosable.
int non_diagnosable();

void test_true() {
  if (non_diagnosable()) {
    // expected-note@-1{{Assuming the condition is true}}
    // expected-note@-2{{Taking true branch}}
    1 / 0;
    // expected-warning@-1{{Division by zero}}
    // expected-note@-2{{Division by zero}}
  }
}

void test_false() {
  if (non_diagnosable()) {
    // expected-note@-1{{Assuming the condition is false}}
    // expected-note@-2{{Taking false branch}}
  } else {
    1 / 0;
    // expected-warning@-1{{Division by zero}}
    // expected-note@-2{{Division by zero}}
  }
}

// Test that we're still reporting that the condition is true,
// when we encounter an exclamation mark (used to be broken).
void test_exclamation_mark() {
  if (!non_diagnosable()) {
    // expected-note@-1{{Assuming the condition is true}}
    // expected-note@-2{{Taking true branch}}
    1 / 0;
    // expected-warning@-1{{Division by zero}}
    // expected-note@-2{{Division by zero}}
  }
}
