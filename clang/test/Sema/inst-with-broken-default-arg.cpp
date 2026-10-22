// RUN: %clang_cc1 %s -verify
// We expect this test to emit a diagnostic, but not crash.
void foo() {
  auto inner_y = [z = 0](auto inner_y, double inner_val = z) {}; // expected-error {{}}
  inner_y(0);
}
