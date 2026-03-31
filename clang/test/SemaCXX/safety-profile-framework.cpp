// RUN: %clang_cc1 -fsyntax-only -verify -fprofiles -std=c++20 %s

// ===================================================================
// Enforce on empty-declaration at TU scope: OK
// ===================================================================
[[profiles::enforce(test::type_cast)]]; // #enforce1

// ===================================================================
// Multiple different profiles enforced: OK
// ===================================================================
[[profiles::enforce(test::bounds)]];

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
// Enforce after a non-empty declaration: error
// ===================================================================
void some_function(); // #some_function
[[profiles::enforce(test::new_profile)]]; // expected-error {{'profiles::enforce' attribute on empty-declaration must precede all non-empty declarations}} \
                                          // expected-note@#some_function {{declaration declared here}}

// ===================================================================
// Enforce inside a namespace: error
// ===================================================================
namespace ns {
  [[profiles::enforce(test::type_cast)]]; // expected-error {{'profiles::enforce' attribute on empty-declaration must be at translation unit scope}}
}

// ===================================================================
// Require not on import: error
// ===================================================================
[[profiles::require(test::type_cast)]]; // expected-error {{'profiles::require' attribute only allowed on module-import-declarations}}

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

// ===================================================================
// Suppress with non-string justification: error
// ===================================================================
[[profiles::suppress(test::type_cast, justification: legacy)]] // expected-error {{'justification' argument of 'profiles::suppress' must be a string literal}}
void bad_justification();

// ===================================================================
// Enforce at block scope: error (this is a null-statement at block
// scope, so enforce cannot appertain to it)
// ===================================================================
void test_block_scope() {
  [[profiles::enforce(test::type_cast)]]; // expected-error {{'profiles::enforce' attribute cannot be applied to a statement}}
}

// ===================================================================
// Diagnostic fires when profile IS enforced
// ===================================================================
void test_enforced_profile_warns() {
  int *p = reinterpret_cast<int*>(0); // expected-warning {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}
