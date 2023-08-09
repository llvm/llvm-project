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
 *
 * WG14 DR448: yes
 * What are the semantics of a # non-directive?
 *
 * WG14 DR454: yes
 * ATOMIC_VAR_INIT (issues 3 and 4)
 *
 * WG14 DR455: yes
 * ATOMIC_VAR_INIT issue 5
 *
 * WG14 DR459: yes
 * atomic_load missing const qualifier
 *
 * WG14 DR475: yes
 * Misleading Atomic library references to atomic types
 *
 * WG14 DR485: yes
 * Problem with the specification of ATOMIC_VAR_INIT
 *
 * WG14 DR486: yes
 * Inconsistent specification for arithmetic on atomic objects
 *
 * WG14 DR490: yes
 * Unwritten Assumptions About if-then
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

/* WG14 DR447: yes
 * Boolean from complex
 *
 * Ensure that the imaginary part contributes to the conversion to bool, not
 * just the real part.
 */
_Static_assert((_Bool)0.0 + 3.0 * (__extension__ 1.0iF), "");  /* c89only-warning {{'_Bool' is a C99 extension}}
                                                                  expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
                                                                */
_Static_assert(!(_Bool)0.0 + 0.0 * (__extension__ 1.0iF), ""); /* c89only-warning {{'_Bool' is a C99 extension}}
                                                                  expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
                                                               */

/* WG14 DR463: yes
 * Left-shifting into the sign bit
 *
 * This DR was NAD and leaves shifting a bit into the high bit of a signed
 * integer type undefined behavior, unlike in C++. Note, the diagnostic is also
 * issued in C++ for shifting into that bit despite being well-defined because
 * the code is questionable and should be validated by the programmer.
 */
void dr463(void) {
  (void)(1 << (__CHAR_BIT__ * sizeof(int))); /* expected-warning {{shift count >= width of type}} */
  (void)(1 << ((__CHAR_BIT__ * sizeof(int)) - 1));
}

/* WG14 DR478: yes
 * Valid uses of the main function
 */
int main(void) {
  /* This DR clarifies that C explicitly allows you to call main() in a hosted
   * environment; it is not special as it is in C++, so recursive calls are
   * fine as well as nonrecursive direct calls.
   */
  main(); /* ok */
}

void dr478(void) {
  int (*fp)(void) = main; /* ok */
  main(); /* ok */
}

/* WG14 DR481: yes
 * Controlling expression of _Generic primary expression
 */
void dr481(void) {
  /* The controlling expression undergoes lvalue to rvalue conversion, and that
   * performs array decay and strips qualifiers.
   */
  (void)_Generic("bla", char *: "blu");
  (void)_Generic((int const){ 0 }, int: "blu");  /* c89only-warning {{compound literals are a C99-specific feature}} */
  (void)_Generic(+(int const){ 0 }, int: "blu"); /* c89only-warning {{compound literals are a C99-specific feature}} */

  (void)_Generic("bla", /* expected-error {{controlling expression type 'char *' not compatible with any generic association type}} */
    char[4]: "blu");    /* expected-warning {{due to lvalue conversion of the controlling expression, association of type 'char[4]' will never be selected because it is of array type}} */

  (void)_Generic((int const){ 0 }, /* expected-error {{controlling expression type 'int' not compatible with any generic association type}}
                                      c89only-warning {{compound literals are a C99-specific feature}}
                                    */
    int const: "blu");             /* expected-warning {{due to lvalue conversion of the controlling expression, association of type 'const int' will never be selected because it is qualified}} */

  (void)_Generic(+(int const){ 0 }, /* expected-error {{controlling expression type 'int' not compatible with any generic association type}}
                                       c89only-warning {{compound literals are a C99-specific feature}}
                                     */
    int const: "blu");              /* expected-warning {{due to lvalue conversion of the controlling expression, association of type 'const int' will never be selected because it is qualified}} */
}

/* WG14 DR489: partial
 * Integer Constant Expression
 *
 * The DR is about whether unevaluated operands have to follow the same
 * restrictions as the rest of the expression in an ICE, and according to the
 * committee, they do.
 */
void dr489(void) {
  struct S {
    int bit : 12 || 1.0f; /* expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}} */
  };
  enum E {
    Val = 0 && 1.0f /* expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}} */
  };

  int i;

  /* FIXME: mentioning the 'aligned' attribute is confusing, but also, should
   * this be folded as an ICE as a GNU extension? GCC does not fold it.
   */
  _Alignas(0 ? i++ : 8) char c; /* expected-error {{'aligned' attribute requires integer constant}} */

  /* FIXME: this should get the constant folding diagnostic as this is not a
   * valid ICE because the floating-point constants are not the immediate
   * operand of a cast. It should then also get a diagnostic about trying to
   * declare a VLA with static storage duration and the C99 extension warning
   * for VLAs in C89.
   */
  static int vla[sizeof(1.0f + 1.0f)];

  int val[5] = { [1 ? 0 : i--] = 12  }; /* expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
                                           c89only-warning {{designated initializers are a C99 feature}}
                                         */

  /* FIXME: this should be the constant folding diagnostic as this is not a
   * valid ICE because of the / operator.
   */
  _Static_assert(sizeof(0 / 0), "");

  /* FIXME: this should also get the constant folding diagnostic as this is not
   * a valid ICE because of the = operator.
   */
  (void)_Generic(i = 12, int : 0); /* expected-warning {{expression with side effects has no effect in an unevaluated context}} */

  switch (i) {
  case (int)0.0f: break;    /* okay, a valid ICE */

  /* FIXME: this should be accepted in C2x and up without a diagnostic, as C23
   * added compound literals to the allowed list of things in an ICE. The
   * diagnostic is correct for C17 and earlier though.
   */
  case (int){ 2 }: break;   /* expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
                               c89only-warning {{compound literals are a C99-specific feature}}
                             */
  case 12 || main(): break; /* expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}} */
  }
}

/* WG14 DR492: yes
 * Named Child struct-union with no Member
 */
struct dr492_t {
  union U11 {  /* expected-warning {{declaration does not declare anything}} */
    int m11;
    float m12;
  };
  int m13;
} dr492;

/* WG14 DR496: yes
 * offsetof questions
 */
void dr496(void) {
  struct A { int n, a [2]; };
  struct B { struct A a; };
  struct C { struct A a[1]; };

  /* Array access & member access expressions are now valid. */
  _Static_assert(__builtin_offsetof(struct B, a.n) == 0, "");
  /* First int below is for 'n' and the second int is for 'a[0]'; this presumes
   * there is no padding involved.
   */
  _Static_assert(__builtin_offsetof(struct B, a.a[1]) == sizeof(int) + sizeof(int), "");

  /* However, we do not support using the -> operator to access a member, even
   * if that would be a valid expression. FIXME: GCC accepts this, perhaps we
   * should as well.
   */
  (void)__builtin_offsetof(struct C, a->n); /* expected-error {{expected ')'}} \
                                               expected-note {{to match this '('}}
                                             */

  /* The DR asked a question about whether defining a new type within offsetof
   * is allowed. C2x N2350 made this explicitly undefined behavior, but GCC and
   * Clang both support it as an extension.
   */
   (void)__builtin_offsetof(struct S { int a; }, a); /* expected-warning{{defining a type within '__builtin_offsetof' is a Clang extension}} */
}

/* WG14 DR499: yes
 * Anonymous structure in union behavior
 */
void dr499(void) {
  union U {
    struct {
      char B1;
      char B2;
      char B3;
      char B4;
    };
    int word;
  } u;

  /* Validate that B1, B2, B3, and B4 do not have overlapping storage, only the
   * anonymous structure and 'word' overlap.
   */
  _Static_assert(__builtin_offsetof(union U, B1) == 0, "");
  _Static_assert(__builtin_offsetof(union U, B2) == 1, "");
  _Static_assert(__builtin_offsetof(union U, B3) == 2, "");
  _Static_assert(__builtin_offsetof(union U, B4) == 3, "");
  _Static_assert(__builtin_offsetof(union U, word) == 0, "");
}
