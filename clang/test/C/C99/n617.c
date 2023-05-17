// RUN: %clang_cc1 -verify %s

// expected-no-diagnostics

/* WG14 N617: yes
 * reliable integer division
 */
void test(void) {
  _Static_assert(59 / 4 == 14, "we didn't truncate properly");
  _Static_assert(59 / -4 == -14, "we didn't truncate properly");
  _Static_assert(-59 / 4 == -14, "we didn't truncate properly");
  _Static_assert(-59 / -4 == 14, "we didn't truncate properly");

  // Might as well test % for the quotient.
  _Static_assert(59 % 4 == 3, "we didn't truncate properly");
  _Static_assert(59 % -4 == 3, "we didn't truncate properly");
  _Static_assert(-59 % 4 == -3, "we didn't truncate properly");
  _Static_assert(-59 % -4 == -3, "we didn't truncate properly");

  // Test the idiom for rounding up.
  _Static_assert((59 + (4 - 1)) / 4 == 15, "failed to 'round up' with the usual idiom");
  _Static_assert((59 + (4 - 1)) % 4 == 2, "failed to 'round up' with the usual idiom");
}

