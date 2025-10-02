// RUN: %clang_cc1 -verify -std=c99 %s
// RUN: %clang_cc1 -verify -std=c23 %s

/* WG14 N448: Partial
 * Restricted pointers
 *
 * NB: we claim partial conformance only because LLVM does not attempt to apply
 * the semantics on local variables or structure data members; it only
 * considers function parameters. However, Clang itself is fully conforming for
 * this feature.
 */

// Restrict is only allowed on pointers.
int * restrict ipr;
int restrict ir; // expected-error {{restrict requires a pointer or reference ('int' is invalid)}}

// Restrict only applies to object pointers.
void (* restrict fp)(void); // expected-error {{pointer to function type 'void (void)' may not be 'restrict' qualified}}

typedef int *int_ptr;
int_ptr restrict ipr2; // okay, still applied to the pointer.

// Show that the qualifer is dropped on lvalue conversion
_Static_assert(
  _Generic(ipr,
    int * : 1,
    int * restrict : 0, // expected-warning {{due to lvalue conversion of the controlling expression, association of type 'int *restrict' will never be selected because it is qualified}}
    default : 0),
  "");

// Show that it's allowed as a qualifier for array parameters.
void f(int array[restrict]) {
  int *ipnr = ipr; // okay to drop the top-level qualifier

  // Show that it's not okay to drop the qualifier when it's not at the top level.
  int * restrict * restrict iprpr;
  int **ipp = iprpr;            // expected-warning {{initializing 'int **' with an expression of type 'int *restrict *restrict' discards qualifiers}}
  int ** restrict ippr = iprpr; // expected-warning {{initializing 'int **restrict' with an expression of type 'int *restrict *restrict' discards qualifiers}}
}

#if __STDC_VERSION__ >= 202311L
// C23 doesn't allow constexpr to mix with restrict. See C23 6.7.2p5.
constexpr int * restrict ip; // expected-error {{constexpr variable cannot have type 'int *const restrict'}}
#endif // __STDC_VERSION__ >= 202311L
