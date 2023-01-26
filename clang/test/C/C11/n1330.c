// RUN: %clang_cc1 -verify -Wgnu-folding-constant %s

/* WG14 N1330: Yes
 * Static assertions
 */

// Test syntactic requirements: first argument must be a constant expression,
// and the second argument must be a string literal. We support the C2x
// extension that allows you to elide the second argument.
int a;
_Static_assert(a, ""); // expected-error {{static assertion expression is not an integral constant expression}}
_Static_assert(1);     // expected-warning {{'_Static_assert' with no message is a C2x extension}}

// Test functional requirements
_Static_assert(1, "this works");
_Static_assert(0, "this fails"); // expected-error {{static assertion failed: this fails}}
_Static_assert(0); // expected-error {{static assertion failed}} \
                      expected-warning {{'_Static_assert' with no message is a C2x extension}}

// Test declaration contexts. We've already demonstrated that file scope works.
struct S {
  _Static_assert(1, "this works");
  union U {
    long l;
    _Static_assert(1, "this works");
  } u;
  enum E {
    _Static_assert(1, "this should not compile"); // expected-error {{expected identifier}}
    One
  } e;
};

void func(                                     // expected-note {{to match this '('}}
  _Static_assert(1, "this should not compile") // expected-error {{expected parameter declarator}} \
                                                  expected-error {{expected ')'}}
);
void func(                                     // expected-note {{to match this '('}}
  _Static_assert(1, "this should not compile") // expected-error {{expected parameter declarator}} \
                                                  expected-error {{expected ')'}}
) {}

void test(void) {
  _Static_assert(1, "this works");
  _Static_assert(0, "this fails"); // expected-error {{static assertion failed: this fails}}

  // The use of a _Static_assert in a for loop declaration is prohibited per
  // 6.8.5p3 requiring the declaration to only declare identifiers for objects
  // having auto or register storage class; a static assertion does not declare
  // an identifier nor an object.
  // FIXME: this diagnostic is pretty terrible.
  int i = 0;
  for (_Static_assert(1, "this should not compile"); i < 10; ++i) // expected-error {{expected identifier or '('}} \
                                                                     expected-error {{expected ';' in 'for' statement specifier}}
    ;

  // Ensure that only an integer constant expression can be used as the
  // controlling expression.
  _Static_assert(1.0f, "this should not compile"); // expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
}

// FIXME: This is using the placeholder date Clang produces for the macro in
// C2x mode; switch to the correct value once it's been published.
#if __STDC_VERSION__ < 202000L
// The use of a _Static_assert in a K&R C function definition is prohibited per
// 6.9.1p6 requiring each declaration to have a declarator (which a static
// assertion does not have) and only declare identifiers from the identifier
// list.
// The error about expecting a ';' is due to the static assertion confusing the
// compiler. It'd be nice if we improved the diagnostics here, but because this
// involves a K&R C declaration, it's low priority.
void knr(a, b, c) // expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C2x}}
  int a, b; // expected-error {{expected ';' at end of declaration}}
  _Static_assert(1, "this should not compile"); // expected-error {{expected identifier or '('}} \
                                                   expected-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
  float c;
{
}
#endif // __STDC_VERSION__
