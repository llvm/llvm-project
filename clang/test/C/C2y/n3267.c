// RUN: %clang_cc1 -std=c2y -verify %s

bool test_if() {
  if (true) {}
  if (bool x = true; x) {}
  if (bool x = false) return x;
  if ([[maybe_unused]] bool x = true) {}
  if (bool x [[maybe_unused]] = true) {}
  if ([[maybe_unused]] int x = 3; x > 0) {}
  return false;
}

int test_switch() {
  int y = 1;
  switch (y) {}

  switch (int x = 1; x) {
  default:
    y += x;
  }

  switch (int x [[maybe_unused]] = 1) {}
  switch ([[maybe_unused]] int x = 1) {}

  switch (int x = 1) {
  default:
    return y + x;
  }
}

bool negative_test_if() {
  if (true; true) {} /* expected-error {{first clause in condition must be a declaration}}
                        expected-warning {{expression result unused}}*/
  if (true; ) {} /* expected-error {{first clause in condition must be a declaration}}
                    expected-error {{expected expression}}
                    expected-warning {{expression result unused}} */
  if (bool x = true; bool y = x) return y; // expected-error {{expected expression}}

  if (true; bool y = true) return y; /* expected-error {{first clause in condition must be a declaration}}
                                        expected-error {{expected expression}}
                                        expected-warning {{expression result unused}}*/
  return false;
}

int negative_test_switch() {
  switch (true; 1) { /* expected-error {{first clause in condition must be a declaration}}
                        expected-warning {{expression result unused}} */
  default:
    break;
  }
  switch (true; ) {} /* expected-error {{first clause in condition must be a declaration}}
                        expected-error {{expected expression}}
                        expected-warning {{expression result unused}} */
  switch (int x = 1; int y = x) { // expected-error {{expected expression}}
  default:
    return y;
  }
}
