// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -analyzer-output=text \
// RUN:   -verify %s

namespace test_tracking_of_lhs_multiplier {
  int f(int x, int y) {
    bool p0 = x < 0;  // expected-note {{Assuming 'x' is >= 0}} \
                      // expected-note {{'p0' initialized to 0}}
    int div = p0 * y; // expected-note {{'div' initialized to 0}}
    return 1 / div;   // expected-note {{Division by zero}} \
                      // expected-warning {{Division by zero}}
  }
} // namespace test_tracking_of_lhs_multiplier

namespace test_tracking_of_rhs_multiplier {
  int f(int x, int y) {
    bool p0 = x < 0;  // expected-note {{Assuming 'x' is >= 0}} \
                      // expected-note {{'p0' initialized to 0}}
    int div = y * p0; // expected-note {{'div' initialized to 0}}
    return 1 / div;   // expected-note {{Division by zero}} \
                      // expected-warning {{Division by zero}}
  }
} // namespace test_tracking_of_rhs_multiplier

namespace test_tracking_of_nested_multiplier {
  int f(int x, int y, int z) {
    bool p0 = x < 0;  // expected-note {{Assuming 'x' is >= 0}} \
                      // expected-note {{'p0' initialized to 0}}
    int div = y*z*p0; // expected-note {{'div' initialized to 0}}
    return 1 / div;   // expected-note {{Division by zero}} \
                      // expected-warning {{Division by zero}}
  }
} // namespace test_tracking_of_nested_multiplier

namespace test_tracking_through_multiple_stmts {
  int f(int x, int y) {
    bool p0 = x < 0;      // expected-note {{Assuming 'x' is >= 0}}
    bool p1 = p0 ? 0 : 1; // expected-note {{'p0' is false}} \
                          // expected-note {{'?' condition is false}}
    bool p2 = 1 - p1;     // expected-note {{'p2' initialized to 0}}
    int div = p2 * y;     // expected-note {{'div' initialized to 0}}
    return 1 / div;       // expected-note {{Division by zero}} \
                          // expected-warning {{Division by zero}}
  }
} // namespace test_tracking_through_multiple_stmts

namespace test_tracking_both_lhs_and_rhs {
  int f(int x, int y) {
    bool p0 = x < 0;   // expected-note {{Assuming 'x' is >= 0}} \
                       // expected-note {{'p0' initialized to 0}}
    bool p1 = y < 0;   // expected-note {{Assuming 'y' is >= 0}} \
                       // expected-note {{'p1' initialized to 0}}
    int div = p0 * p1; // expected-note {{'div' initialized to 0}}
    return 1 / div;    // expected-note {{Division by zero}} \
                       // expected-warning {{Division by zero}}
  }
} // namespace test_tracking_both_lhs_and_rhs

namespace test_tracking_of_multiplier_and_parens {
  int f(int x, int y, int z) {
    bool p0 = x < 0;    // expected-note {{Assuming 'x' is >= 0}} \
                        // expected-note {{'p0' initialized to 0}}
    int div = y*(z*p0); // expected-note {{'div' initialized to 0}}
    return 1 / div;     // expected-note {{Division by zero}} \
                        // expected-warning {{Division by zero}}
  }
} // namespace test_tracking_of_multiplier_and_parens

namespace test_tracking_of_divisible {
  int f(int x, int y) {
    bool p0 = x < 0;    // expected-note {{Assuming 'x' is >= 0}} \
                        // expected-note {{'p0' initialized to 0}}
    int div = p0 / y;   // expected-note {{'div' initialized to 0}}
    return 1 / div;     // expected-note {{Division by zero}} \
                        // expected-warning {{Division by zero}}
  }
} // namespace test_tracking_of_divisible

namespace test_tracking_of_modulo {
  int f(int x, int y) {
    bool p0 = x < 0;    // expected-note {{Assuming 'x' is >= 0}} \
                        // expected-note {{'p0' initialized to 0}}
    int div = p0 % y;   // expected-note {{'div' initialized to 0}}
    return 1 / div;     // expected-note {{Division by zero}} \
                        // expected-warning {{Division by zero}}
  }
} // namespace test_tracking_of_modulo

namespace test_tracking_of_assignment {
  int f(int x) {
    bool p0 = x < 0;    // expected-note {{Assuming 'x' is >= 0}} \
                        // expected-note {{'p0' initialized to 0}}
    int div = 1;
    div *= p0;          // expected-note {{The value 0 is assigned to 'div'}}
    return 1 / div;     // expected-note {{Division by zero}} \
                        // expected-warning {{Division by zero}}
  }
} // namespace test_tracking_of_assignment
