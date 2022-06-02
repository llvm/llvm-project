/* RUN: %clang_cc1 -std=c89 -verify=expected,c89only -pedantic -Wno-declaration-after-statement -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c89 -verify=expected,c89only -pedantic -Wno-declaration-after-statement -Wno-c11-extensions -fno-signed-char %s
   RUN: %clang_cc1 -std=c99 -verify=expected,c99untilc2x -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -verify=expected,c99untilc2x -pedantic %s
   RUN: %clang_cc1 -std=c17 -verify=expected,c99untilc2x -pedantic %s
   RUN: %clang_cc1 -std=c2x -verify=expected,c2xandup -pedantic %s
 */

/* The following are DRs which do not require tests to demonstrate
 * conformance or nonconformance.
 *
 * WG14 DR001: yes
 * Do functions return values by copying?
 *
 * WG14 DR005: yes
 * May a conforming implementation define and recognize a pragma which would
 * change the semantics of the language?
 *
 * WG14 DR008: yes
 * Can a conforming C compiler to perform dead-store elimination?
 *
 * WG14 DR020: yes
 * Is a compiler which allows the Relaxed Ref/Def linkage model to be
 * considered a conforming compiler?
 *
 * WG14 DR025: yes
 * What is meant by 'representable floating-point value?'
 *
 * WG14 DR026: yes
 * Can a strictly conforming program contain a string literal with '$' or '@'?
 *
 * WG14 DR033: yes
 * Conformance questions around 'shall' violations outside of constraints
 * sections
 *
 * WG14 DR036: yes
 * May floating-point constants be represented with more precision than implied
 * by its type?
 *
 * WG14 DR037: yes
 * Questions about multibyte characters and Unicode
 *
 * WG14 DR051: yes
 * Question on pointer arithmetic
 *
 * WG14 DR052: yes
 * Editorial corrections
 *
 * WG14 DR056: yes
 * Floating-point representation precision requirements
 *
 * WG14 DR057: yes
 * Is there an integral type for every pointer?
 *
 * WG14 DR059: yes
 * Do types have to be completed?
 *
 * WG14 DR063: dup 056
 * Floating-point representation precision requirements
 *
 * WG14 DR067: yes
 * Integer and integral type confusion
 *
 * WG14 DR069: yes
 * Questions about the representation of integer types
 *
 * WG14 DR077: yes
 * Stability of addresses
 *
 * WG14 DR080: yes
 * Merging of string constants
 *
 * WG14 DR086: yes
 * Object-like macros in system headers
 *
 * WG14 DR091: yes
 * Multibyte encodings
 *
 * WG14 DR092: dup 060
 * Partial initialization of strings
 *
 * WG14 DR093: yes
 * Reservation of identifiers
 */


/* WG14 DR004: yes
 * Are multiple definitions of unused identifiers with external linkage
 * permitted?
 */
int dr004(void) {return 0;} /* expected-note {{previous definition is here}} */
int dr004(void) {return 1;} /* expected-error {{redefinition of 'dr004'}} */

/* WG14 DR007: yes
 * Are declarations of the form struct-or-union identifier ; permitted after
 * the identifier tag has already been declared?
 */
struct dr007_a;
struct dr007_a;
struct dr007_a {int a;};
struct dr007_a;
struct dr007_b {int a;};
struct dr007_b;


/* WG14 DR009: no
 * Use of typedef names in parameter declarations
 *
 * FIXME: This should be diagnosed as expecting a declaration specifier instead
 * of treated as declaring a parameter of type 'int (*)(dr009_t);'
 */
typedef int dr009_t;
void dr009_f((dr009_t)); /* c99untilc2x-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}
                            c2xandup-error {{a type specifier is required for all declarations}} */

/* WG14 DR010:
 * Is a typedef to an incomplete type legal?
 */
typedef int dr010_t[];
dr010_t dr010_a = {1};
dr010_t dr010_b = {1, 2};
int dr010_c = sizeof(dr010_t); /* expected-error {{invalid application of 'sizeof' to an incomplete type 'dr010_t' (aka 'int[]')}} */

/* WG14 DR011: yes
 * Merging of declarations for linked identifier
 *
 * Note: more of this DR is tested in dr011.c
 *
 * WG14 DR034: yes
 * External declarations in different scopes
 *
 * Note: DR034 has a question resolved by DR011 and another question where the
 * result is UB.
 */
static int dr011_a[]; /* expected-warning {{tentative array definition assumed to have one element}} */
void dr011(void) {
  extern int i[];
  {
    /* a different declaration of the same object */
    extern int i[10];
    (void)sizeof(i);
    _Static_assert(sizeof(i) == 10 * sizeof(int), "fail");
  }
  (void)sizeof(i); /* expected-error {{invalid application of 'sizeof' to an incomplete type 'int[]'}} */

  extern int dr011_a[10];
  (void)sizeof(dr011_a);
  _Static_assert(sizeof(dr011_a) == 10 * sizeof(int), "fail");

  extern int j[10];
  {
    extern int j[];
    (void)sizeof(j);
    _Static_assert(sizeof(j) == 10 * sizeof(int), "fail");
  }
}

/* WG14 DR012: yes
 * Is it valid to take the address of a dereferenced void pointer?
 */
void dr012(void *p) {
  /* The behavior changed between C89 and C99. */
  (void)&*p; /* c89only-warning {{ISO C forbids taking the address of an expression of type 'void'}} */
}

/* WG14 DR013: yes
 * Compatible and composite function types
 */
int dr013(int a[4]);
int dr013(int a[5]);
int dr013(int *a);

struct dr013_t {
struct dr013_t *p;
} dr013_v[sizeof(struct dr013_t)];

/* WG14 DR015: yes
 * What is the promoted type of a plain int bit-field?
 */
void dr015(void) {
  struct S {
    int small_int_bitfield : 16;
    unsigned int small_uint_bitfield : 16;
    int int_bitfield : 32;
    unsigned int uint_bitfield : 32;
  } s;
  _Static_assert(__builtin_types_compatible_p(__typeof__(+s.small_int_bitfield), int), "fail");
  _Static_assert(__builtin_types_compatible_p(__typeof__(+s.small_uint_bitfield), int), "fail");
  _Static_assert(__builtin_types_compatible_p(__typeof__(+s.int_bitfield), int), "fail");
  _Static_assert(__builtin_types_compatible_p(__typeof__(+s.uint_bitfield), unsigned int), "fail");
}

/* WG14 DR027: yes
 * Can there be characters in the character set that are not in the required
 * source character set?
 */
#define THIS$AND$THAT(a, b) ((a) + (b)) /* expected-warning 2 {{'$' in identifier}} */
_Static_assert(THIS$AND$THAT(1, 1) == 2, "fail"); /* expected-warning 2 {{'$' in identifier}} */


/* WG14 DR029: no
 * Do two types have to have the same tag to be compatible?
 * Note: the rule changed in C99 to be different than the resolution to DR029,
 * so it's not clear there's value in implementing this DR.
 */
_Static_assert(__builtin_types_compatible_p(struct S { int a; }, union U { int a; }), "fail"); /* expected-error {{static_assert failed due to requirement '__builtin_types_compatible_p(struct S, union U)' "fail"}} */

/* WG14 DR031: yes
 * Can constant expressions overflow?
 */
void dr031(int i) {
  switch (i) {
  case __INT_MAX__ + 1: break; /* expected-warning {{overflow in expression; result is -2147483648 with type 'int'}} */
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wswitch"
  /* Silence the targets which issue:
   * warning: overflow converting case value to switch condition type (2147483649 to 18446744071562067969)
   */
  case __INT_MAX__ + 2ul: break;
  #pragma clang diagnostic pop
  case (__INT_MAX__ * 4) / 4: break; /* expected-warning {{overflow in expression; result is -4 with type 'int'}} */
  }
}

/* WG14 DR032: no
 * Must implementations diagnose extensions to the constant evaluation rules?
 *
 * This should issue a diagnostic because a constant-expression is a
 * conditional-expression, which excludes the comma operator.
 */
int dr032 = (1, 2); /* expected-warning {{left operand of comma operator has no effect}} */

#if __STDC_VERSION__ < 202000L
/* WG14 DR035: partial
 * Questions about definition of functions without a prototype
 */
void dr035_1(a, b) /* expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C2x}} */
  int a(enum b {x, y}); /* expected-warning {{declaration of 'enum b' will not be visible outside of this function}} */
  int b; {
  int test = x; /* expected-error {{use of undeclared identifier 'x'}} */
}

void dr035_2(c) /* expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C2x}} */
  enum m{q, r} c; { /* expected-warning {{declaration of 'enum m' will not be visible outside of this function}} */
  /* FIXME: This should be accepted because the scope of m, q, and r ends at
   * the closing brace of the function per C89 6.1.2.1.
   */
  int test = q; /* expected-error {{use of undeclared identifier 'q'}} */
}
#endif /* __STDC_VERSION__ < 202000L */

/* WG14 DR038: yes
 * Questions about argument substitution during macro expansion
 */
#define DR038_X 0x000E
#define DR038_Y 0x0100
#define DR038(a) a
_Static_assert(DR038(DR038_X + DR038_Y) == DR038_X + DR038_Y, "fail");

/* WG14 DR039: yes
 * Questions about the "C" locale
 */
_Static_assert(sizeof('a') == sizeof(int), "fail");

/* WG14 DR040: partial
 * 9 unrelated questions about C89
 *
 * Question 6
 */
struct dr040 { /* expected-note {{definition of 'struct dr040' is not complete until the closing '}'}} */
  char c;
  short s;
  int i[__builtin_offsetof(struct dr040, s)]; /* expected-error {{offsetof of incomplete type 'struct dr040'}} */
};

/* WG14 DR043: yes
 * On the definition of the NULL macro
 */
void dr043(void) {
  #include <stddef.h>
  /* NULL has to be an integer constant expression with the value 0, or such an
   * expression cast to void *. If it's an integer constant expression other
   * than the literal 0 (such as #define NULL 4-4), this would fail to compile
   * unless the macro replacement list is properly parenthesized as it would
   * expand to: (void)(void *)4-4;
   */
   (void)(void *)NULL;

  /* If the NULL macro is an integer constant expression with the value 0 and
   * it has been cast to void *, ensure that it's also fully parenthesized. If
   * it isn't (such as #define NULL (void *)0), this would fail to compile as
   * would expand to (void *)0->a; which gives a diagnostic about int not being
   * a pointer, instead of((void *)0)->a; which gives a diagnostic about the
   * base reference being void and not a structure.
   */
   NULL->a; /* expected-error {{member reference base type 'void' is not a structure or union}} */
}

/* WG14 DR044: yes
 * On the result of the offsetof macro
 */
void dr044(void) {
  #include <stddef.h>
  struct S { int a, b; };
  /* Ensure that the result of offsetof is usable in a constant expression. */
  _Static_assert(offsetof(struct S, b) == sizeof(int), "fail");
}

/* WG14 DR046: yes
 * Use of typedef names in parameter declarations
 */
typedef int dr046_t;
int dr046(int dr046_t) { return dr046_t; }

/* WG14 DR047: yes
 * Questions about declaration conformance
 */
struct dr047_t; /* expected-note 2 {{forward declaration of 'struct dr047_t'}} */
struct dr047_t *dr047_1(struct dr047_t *p) {return p; }
struct dr047_t *dr047_2(struct dr047_t a[]) {return a; } /* expected-error {{array has incomplete element type 'struct dr047_t'}} */
int *dr047_3(int a2[][]) {return *a2; } /* expected-error {{array has incomplete element type 'int[]'}} */
extern struct dr047_t es1;
extern struct dr047_t es2[1]; /* expected-error {{array has incomplete element type 'struct dr047_t'}} */

/* WG14 DR050: yes
 * Do wide string literals implicitly include <stddef.h>?
 */
void dr050(void) {
  /* The NULL macro is previously defined because we include <stddef.h> for
   * other tests. Undefine the macro to demonstrate that use of a wide string
   * literal doesn't magically include the header file.
   */
  #undef NULL
  (void)L"huttah!";
  (void)NULL; /* expected-error {{use of undeclared identifier 'NULL'}} */
}

#if __STDC_VERSION__ < 202000L
/* WG14 DR053: yes
 * Accessing a pointer to a function with a prototype through a pointer to
 * pointer to function without a prototype
 */
void dr053(void) {
  int f(int);
  int (*fp1)(int);
  int (*fp2)();  /* expected-warning {{a function declaration without a prototype is deprecated in all versions of C}} */
  int (**fpp)(); /* expected-warning {{a function declaration without a prototype is deprecated in all versions of C}} */

  fp1 = f;
  fp2 = fp1;
  (*fp2)(3);  /* expected-warning {{passing arguments to a function without a prototype is deprecated in all versions of C and is not supported in C2x}} */
  fpp = &fp1;
  (**fpp)(3); /* expected-warning {{passing arguments to a function without a prototype is deprecated in all versions of C and is not supported in C2x}} */
}
#endif /* __STDC_VERSION__ < 202000L */

/* WG14 DR064: yes
 * Null pointer constants
 */
char *dr064_1(int i, int *pi) {
  *pi = i;
  return 0;
}

char *dr064_2(int i, int *pi) {
  return (*pi = i, 0); /* expected-warning {{incompatible integer to pointer conversion returning 'int' from a function with result type 'char *'}} */
}

/* WG14 DR068: yes
 * 'char' and signed vs unsigned integer types
 */
void dr068(void) {
  #include <limits.h>

#if CHAR_MAX == SCHAR_MAX
  /* char is signed */
  _Static_assert('\xFF' == -1, "fail");
#else
  /* char is unsigned */
  _Static_assert('\xFF' == 0xFF, "fail");
#endif
}

#if __STDC_VERSION__ < 202000L
/* WG14: DR070: yes
 * Interchangeability of function arguments
 *
 * Note: we could issue a pedantic warning in this case. We are claiming
 * conformance not because we diagnose the UB when we could but because we're
 * not obligated to do anything about it and we make it "just work" via the
 * usual conversion rules.
 *
 * This behavior is specific to functions without prototypes. A function with
 * a prototype causes implicit conversions rather than relying on default
 * argument promotion and warm thoughts.
 */
void dr070_1(c) /* expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C2x}} */
  int c; {
}

void dr070_2(void) {
  dr070_1(6);
  dr070_1(6U); /* Pedantically UB */
}
#endif /* __STDC_VERSION__ < 202000L */

/* WG14 DR071: yes
 * Enumerated types
 */
enum dr071_t { foo_A = 0, foo_B = 1, foo_C = 8 };
void dr071(void) {
  /* Test that in-range values not present in the enumeration still round-trip
   * to the original value.
   */
  _Static_assert(100 == (int)(enum dr071_t)100, "fail");
}

/* WG14 DR081: yes
 * Left shift operator
 */
void dr081(void) {
  /* Demonstrate that we don't crash when left shifting a signed value; that's
   * implementation defined behavior.
   */
 _Static_assert(-1 << 1 == -2, "fail"); /* Didn't shift a zero into the "sign bit". */
 _Static_assert(1 << 3 == 1u << 3u, "fail"); /* Shift of a positive signed value does sensible things. */
}

/* WG14 DR084: yes
 * Incomplete type in function declaration
 *
 * Note: because the situation is UB, we're free to do what we want. We elect
 * to accept and require the incomplete type to be completed before the
 * function definition.
 */
struct dr084_t; /* expected-note {{forward declaration of 'struct dr084_t'}} */
extern void (*dr084_1)(struct dr084_t);
void dr084_2(struct dr084_t);
void dr084_2(struct dr084_t val) {} /* expected-error {{variable has incomplete type 'struct dr084_t'}} */

/* WG14 DR088: yes
 * Compatibility of incomplete types
 */
struct dr088_t_1;

void dr088_f(struct dr088_t_1 *); /* expected-note {{passing argument to parameter here}} */
void dr088_1(void) {
  /* Distinct type from the file scope forward declaration. */
  struct dr088_t_1;
  /* FIXME: this diagnostic could be improved to not be utterly baffling. */
  dr088_f((struct dr088_t_1 *)0); /* expected-warning {{incompatible pointer types passing 'struct dr088_t_1 *' to parameter of type 'struct dr088_t_1 *'}} */
}

void dr088_2(struct dr088_t_1 *p) { /* Pointer to incomplete type. */ }
struct dr088_t_1 { int i; }; /* Type is completed. */
void dr088_3(struct dr088_t_1 s) {
  /* When passing a pointer to the completed type, is it the same type as the
   * incomplete type used in the call declaration?
   */
  dr088_2(&s);
}

/* WG14 DR089: yes
 * Multiple definitions of macros
 */
#define DR089 object_like             /* expected-note {{previous definition is here}} */
#define DR089(argument) function_like /* expected-warning {{'DR089' macro redefined}} */

/* WG14 DR095: yes
 * Is initialization as constrained as assignment?
 */
void dr095(void) {
  /* Ensure that type compatibility constraints on assignment are also honored
   * for initializations.
   */
  struct One {
    int a;
  } one;
  struct Two {
    float f;
  } two = one; /* expected-error {{initializing 'struct Two' with an expression of incompatible type 'struct One'}} */

  two = one; /* expected-error {{assigning to 'struct Two' from incompatible type 'struct One'}} */
}

/* WG14 DR096: yes
 * Arrays of incomplete types
 */
void dr096(void) {
  typedef void func_type(void);
  func_type array_funcs[10]; /* expected-error {{'array_funcs' declared as array of functions of type 'func_type' (aka 'void (void)')}} */

  void array_void[10]; /* expected-error {{array has incomplete element type 'void'}} */

  struct S; /* expected-note {{forward declaration of 'struct S'}} */
  struct S s[10]; /* expected-error {{array has incomplete element type 'struct S'}} */

  union U; /* expected-note {{forward declaration of 'union U'}} */
  union U u[10]; /* expected-error {{array has incomplete element type 'union U'}} */
  union U { int i; };

  int never_completed_incomplete_array[][]; /* expected-error {{array has incomplete element type 'int[]'}} */

  extern int completed_later[][]; /* expected-error {{array has incomplete element type 'int[]'}} */
  extern int completed_later[10][10];
}

/* WG14 DR098: yes
 * Pre/post increment/decrement of function or incomplete types
 */
void dr098(void) {
  typedef void func_type(void);
  func_type fp;
  struct incomplete *incomplete_ptr;

  ++fp; /* expected-error {{cannot increment value of type 'func_type' (aka 'void (void)')}} */
  fp++; /* expected-error {{cannot increment value of type 'func_type' (aka 'void (void)')}} */
  --fp; /* expected-error {{cannot decrement value of type 'func_type' (aka 'void (void)')}} */
  fp--; /* expected-error {{cannot decrement value of type 'func_type' (aka 'void (void)')}} */

  (*incomplete_ptr)++; /* expected-error {{cannot increment value of type 'struct incomplete'}} */
  ++(*incomplete_ptr); /* expected-error {{cannot increment value of type 'struct incomplete'}} */
  (*incomplete_ptr)--; /* expected-error {{cannot decrement value of type 'struct incomplete'}} */
  --(*incomplete_ptr); /* expected-error {{cannot decrement value of type 'struct incomplete'}} */
}
