/* RUN: %clang_cc1 -std=c89 -verify=expected,c89only -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -verify=expected -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -verify=expected -pedantic %s
   RUN: %clang_cc1 -std=c17 -verify=expected -pedantic %s
   RUN: %clang_cc1 -std=c2x -verify=expected -pedantic %s
 */

/* The following are DRs which do not require tests to demonstrate
 * conformance or nonconformance.
 *
 * WG14 DR401: yes
 * "happens before" can not be cyclic
 *
 * WG14 DR402: yes
 * Memory model coherence is not aligned with C++11
 *
 * WG14 DR404: yes
 * Joke fragment remains in a footnote
 *
 * WG14 DR406: yes
 * Visible sequences of side effects are redundant
 *
 * WG14 DR415: yes
 * Missing divide by zero entry in Annex J
 *
 * WG14 DR417: yes
 * Annex J not updated with necessary aligned_alloc entries
 *
 * WG14 DR419: yes
 * Generic Functions
 *
 * WG14 DR420: yes
 * Sytax error in specification of for-statement
 *
 * WG14 DR425: yes
 * No specification for the access to variables with temporary lifetime
 *
 * WG14 DR434: yes
 * Possible defect report: Missing constraint w.r.t. Atomic
 *
 * WG14 DR435: yes
 * Possible defect report: Missing constraint w.r.t. Imaginary
 *
 * WG14 DR436: yes
 * Request for interpretation of C11 6.8.5#6
 * Note: This is not really testable because it requires -O1 or higher for LLVM
 * to perform its reachability analysis and -Wunreachable-code only verifies
 * diagnostic behavior, not runtime behavior. Also, both are a matter of QoI as
 * to what they optimize/diagnose. But if someone thinks of a way to test this,
 * we can add a test case for it then.
 */

/* WG14 DR412: yes
 * #elif
 *
 * Note: this is testing that #elif behaves the same as #else followed by #if.
 */
#if 1
#elif this is not a valid expression
#else
  #if this is not a valid expression
  #endif
#endif

/* WG14 DR413: yes
 * Initialization
 */
void dr413(void) {
  typedef struct {
    int k;
    int l;
    int a[2];
  } T;

  typedef struct {
    int i;
    T t;
  } S;

  /* Ensure that explicit initialization (.t = { ... }) takes precedence over a
   * later implicit partial initialization (.t.l = 41). The value should be 42,
   * not 0.
   */
  _Static_assert((S){ /* c89only-warning {{compound literals are a C99-specific feature}}
                         expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
                       */
      1,
      .t = {          /* c89only-warning {{designated initializers are a C99 feature}} */
        .l = 43,      /* c89only-warning {{designated initializers are a C99 feature}}
                         expected-note {{previous initialization is here}}
                       */
        .k = 42,
        .a[1] = 19,   /* expected-note {{previous initialization is here}} */
        .a[0] = 18
      },
      .t.l = 41,      /* expected-warning {{initializer overrides prior initialization of this subobject}} */
      .t.a[1] = 17    /* expected-warning {{initializer overrides prior initialization of this subobject}} */
    }.t.k == 42, "");
}

/* WG14 DR423: partial
 * Defect Report relative to n1570: underspecification for qualified rvalues
 */

/* FIXME: this should pass because the qualifier on the return type should be
 * dropped when forming the function type.
 */
const int dr423_const(void);
int dr423_nonconst(void);
_Static_assert(__builtin_types_compatible_p(__typeof__(dr423_const), __typeof__(dr423_nonconst)), "fail"); /* expected-error {{fail}} */

void dr423_func(void) {
  const int i = 12;
  __typeof__(i) v1 = 12; /* expected-note {{variable 'v1' declared const here}} */
  __typeof__((const int)12) v2 = 12;

  v1 = 100; /* expected-error {{cannot assign to variable 'v1' with const-qualified type 'typeof (i)' (aka 'const int')}} */
  v2 = 100; /* Not an error; the qualifier was stripped. */
}

/* WG14 DR432: yes
 * Possible defect report: Is 0.0 required to be a representable value?
 *
 * We're going to lean on the fpclassify builtin to tell us whether 0.0
 * represents the value 0, and we'll test that adding and subtracting 0.0 does
 * not change the value, and we'll hope that's enough to validate this DR.
 */
_Static_assert(__builtin_fpclassify(0, 1, 2, 3, 4, 0.0f) == 4, "");
_Static_assert((1.0 / 3.0) + 0.0 == (1.0 / 3.0) - 0.0, ""); /* expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}} */

/* WG14 DR444: partial
 * Issues with alignment in C11, part 1
 */
void dr444(void) {
  _Alignas(int) int i;
   _Alignas(int) struct S {
    _Alignas(int) int i;
  } s;

  /* FIXME: This should be accepted as per this DR. */
  int j = (_Alignas(int) int){12}; /* expected-error {{expected expression}} */

 /* FIXME: The diagnostic in this case is really bad; moving the specifier to
  * where the diagnostic recommends causes a different, more inscrutable error
  * about anonymous structures.
  */
  _Alignas(int) struct T { /* expected-warning {{attribute '_Alignas' is ignored, place it after "struct" to apply attribute to type declaration}} */
    int i;
  };

  struct U {
    _Alignas(int) int bit : 1; /* expected-error {{'_Alignas' attribute cannot be applied to a bit-field}} */
  };

  _Alignas(int) typedef int foo;  /* expected-error {{'_Alignas' attribute only applies to variables and fields}} */
  _Alignas(int) register int bar; /* expected-error {{'_Alignas' attribute cannot be applied to a variable with 'register' storage class}} */
  _Alignas(int) void func(void);  /* expected-error {{'_Alignas' attribute only applies to variables and fields}} */

  /* FIXME: it is correct for us to accept this per 6.7.3p5, but it seems like
   * a situation we'd want to diagnose because the alignments are different and
   * the user probably doesn't know which one "wins".
   */
  _Alignas(int) _Alignas(double) int k;
}
