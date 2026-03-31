// RUN: %clang_cc1 -fsyntax-only -verify -fprofiles -std=c++20 %s

// ===================================================================
// Enforce on empty-declaration at TU scope: OK
// ===================================================================
[[profiles::enforce(test::type_cast)]]; // #enforce1

// ===================================================================
// Enforce: exact repetition is OK
// ===================================================================
[[profiles::enforce(test::type_cast)]];

// ===================================================================
// Enforce: mismatch is an error
// ===================================================================
[[profiles::enforce(test::type_cast(strict: true))]]; // expected-error {{repeated enforcement of profile 'test::type_cast' with different designator}} \
                                                      // expected-note@#enforce1 {{previous attribute is here}}

// ===================================================================
// Suppress on declarations
// ===================================================================
[[profiles::suppress(test::type_cast)]]
int suppressed_var;

[[profiles::suppress(test::type_cast)]]
void suppressed_func();

// ===================================================================
// Suppress on statements
// ===================================================================
void test_stmt_suppress() {
  [[profiles::suppress(test::type_cast)]] int x = 0;
  [[profiles::suppress(test::type_cast)]] { int y = 0; }
}
