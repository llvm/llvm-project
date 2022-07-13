/* RUN: %clang_cc1 -std=c89 -fsyntax-only -verify=expected,c89only -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -triple x86_64-unknown-linux -fsyntax-only -verify=expected,c99untilc2x -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -triple x86_64-unknown-win32 -fms-compatibility -fsyntax-only -verify=expected,c99untilc2x -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -fsyntax-only -verify=expected,c99untilc2x -pedantic %s
   RUN: %clang_cc1 -std=c17 -fsyntax-only -verify=expected,c99untilc2x -pedantic %s
   RUN: %clang_cc1 -std=c2x -fsyntax-only -verify=expected,c2xandup -pedantic %s
 */

/* The following are DRs which do not require tests to demonstrate
 * conformance or nonconformance.
 *
 * WG14 DR201: yes
 * Integer types longer than long
 *
 * WG14 DR211: yes
 * Accuracy of decimal string to/from "binary" (non-decimal) floating-point conversions
 *
 * WG14 DR215: yes
 * Equality operators
 *
 * WG14 DR218: yes
 * Signs of non-numeric floating point values
 *
 * WG14 DR219: yes
 * Effective types
 *
 * WG14 DR221: yes
 * Lacuna in pointer arithmetic
 *
 * WG14 DR222: yes
 * Partially initialized structures
 *
 * WG14 DR234: yes
 * Miscellaneous Typos
 *
 * WG14 DR245: yes
 * Missing paragraph numbers
 *
 * WG14 DR247: yes
 * Are values a form of behaviour?
 *
 * WG14 DR248: yes
 * Limits are required for optional types
 *
 * WG14 DR255: yes
 * Non-prototyped function calls and argument mismatches
 *
 * WG14 DR267: yes
 * Typos in 5.1.2.3, 7.24.4.4.5, 7.24.6.1, 7.24.6.1
 *
 * WG14 DR273: yes
 * Meaning of __STDC_ISO_10646__
 *
 * WG14 DR278: yes
 * Lacuna in character encodings
 *
 * WG14 DR279: yes
 * Wide character code values for members of the basic character set
 *
 * WG14 DR282: yes
 * Flexible array members & struct padding
 *
 * WG14 DR292: yes
 * Use of the word variable
 */


/* WG14 DR204: yes
 * size_t and ptrdiff_t as a long long type
 */
void dr204(void) {
  __typeof__(sizeof(0)) s;
  __typeof__((int *)0 - (int *)0) p;
  signed long sl;
#if __LLONG_WIDTH__ > __LONG_WIDTH__
  /* If the implementation supports a standard integer type larger than signed
   * long, it's okay for size_t and ptrdiff_t to have a greater integer
   * conversion rank than signed long.
   *
   * Note, it's not required that the implementation use that larger conversion
   * rank; it's acceptable to use an unsigned long or unsigned int for the size
   * type (those ranks are not greater than that of signed long).
   */
   (void)_Generic(s + sl, unsigned long long : 1, unsigned long : 1, unsigned int : 1); /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */
   (void)_Generic(p + sl, signed long long : 1, signed long : 1, signed int : 1);       /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */
#elif __LLONG_WIDTH__ == __LONG_WIDTH__
  /* But if the implementation doesn't support a larger standard integer type
   * than signed long, the conversion rank should prefer signed long if the type
   * is signed (ptrdiff_t) or unsigned long if the type is unsigned (size_t).
   *
   * Note, as above, unsigned/signed int is also acceptable due to having a
   * lesser integer conversion rank.
   */
   (void)_Generic(s + sl, unsigned long : 1, unsigned int : 1);
   (void)_Generic(p + sl, signed long : 1, signed int : 1);
#else
#error "Something has gone off the rails"
#endif
}

/* WG14 DR207: partial
 * Handling of imaginary types
 *
 * FIXME: Clang recognizes the _Imaginary keyword but does not support the data
 * type.
 */
void dr207(void) {
  _Imaginary float f; /* expected-error {{imaginary types are not supported}}
                         c89only-warning {{'_Imaginary' is a C99 extension}}
                       */
}

/* WG14 DR216: yes
 * Source character encodings
 */
void dr216(void) {
#define A(x) _Static_assert((char)x >= 0, "no")
  A('A'); A('B'); A('C'); A('D'); A('E'); A('F'); A('G'); A('H'); A('I');
  A('J'); A('K'); A('L'); A('M'); A('N'); A('O'); A('P'); A('Q'); A('R');
  A('S'); A('T'); A('U'); A('V'); A('W'); A('X'); A('Y'); A('Z');

  A('a'); A('b'); A('c'); A('d'); A('e'); A('f'); A('g'); A('h'); A('i');
  A('j'); A('k'); A('l'); A('m'); A('n'); A('o'); A('p'); A('q'); A('r');
  A('s'); A('t'); A('u'); A('v'); A('w'); A('x'); A('y'); A('z');

  A('0'); A('1'); A('2'); A('3'); A('4');
  A('5'); A('6'); A('7'); A('8'); A('9');

  A('!'); A('"'); A('#'); A('%'); A('&'); A('\''); A('('); A(')'); A('*');
  A('+'); A(','); A('-'); A('.'); A('/'); A(':'); A(';'); A('<'); A('=');
  A('>'); A('?'); A('['); A('\\'); A(']'); A('^'); A('_'); A('{'); A('|');
  A('}'); A('~');
#undef A
}

/* WG14 DR230: yes
 * Enumerated type rank
 */
void dr230(void) {
  enum E {
    Value = __INT_MAX__
  } e;
  /* The enumeration type has a compatible type that is a signed or unsigned
   * integer type, or char. But it has to be large enough to hold all of the
   * values of the enumerators. So it needs to be at least int or unsigned int.
   *
   * The integer conversion rank for an enumeration is the same as its
   * compatible type (C99 6.3.1.1p1), so it's eligible for integer promotions
   * to either int or unsigned int, depending on the compatible type
   * (C99 6.3.1.1p2).
   */
  (void)_Generic(e, int : 1, unsigned int : 1);
  (void)_Generic((enum E)Value, int : 1, unsigned int : 1);
  /* The enumerators themselves have type int (C99 6.7.2.2p3). */
  (void)_Generic(Value, int : 1);
}

/* WG14 DR231: no
 * Semantics of text-line and non-directive
 *
 * One of the preprocessing groups to support is # non-directive (C99 6.10p1),
 * which is defined as pp-tokens followed by a newline. However, we fail to
 * translate the program if we don't recognize the directive, and we don't take
 * note when what follows the # is not a valid preprocessing token.
 */

/* FIXME: this should not fail. */
# nope /* expected-error {{invalid preprocessing directive}} */

/* FIXME: this should fail, but not because of the unknown directive; it should
 * fail because of the invalid preprocessing-token.
 */
# 'a
/* expected-error@-1 {{invalid preprocessing directive}} \
   expected-warning@-1 {{missing terminating ' character}}
*/

/* WG14 DR237: no
 * Declarations using [static]
 */
void dr237_f(int array[static 10]); /* c89only-warning {{static array size is a C99 feature}}
                                       expected-note {{callee declares array parameter as static here}}
                                     */
void dr237_1(void) {
  int array[4];
  dr237_f(array); /* expected-warning {{array argument is too small; contains 4 elements, callee requires at least 10}} */
}

/* FIXME: the composite type for this declaration should retain the static
 * array extent instead of losing it.
 */
void dr237_f(int array[]);

void dr237_2(void) {
  int array[4];
  /* FIXME: this should diagnose the same as above. */
  dr237_f(array);
}

/* WG14 DR246: yes
 * Completion of declarators
 */
void dr246(void) {
  int i[i]; /* expected-error {{use of undeclared identifier 'i'}} */
}

/* WG14 DR250: yes
 * Non-directives within macro arguments
 */
void dr250(void) {
#define dr250_nothing(x)

  /* FIXME: See DR231 regarding the error about an invalid preprocessing
   * directive.
   */

  dr250_nothing(
#nondirective    /* expected-error {{invalid preprocessing directive}}
                    expected-warning {{embedding a directive within macro arguments has undefined behavior}}
                  */
  )

#undef dr250_nothing
}

/* WG14 DR251: yes
 * Are struct fred and union fred the same type?
 */
union dr251_fred { int a; }; /* expected-note {{previous use is here}} */
void dr251(void) {
  struct dr251_fred *ptr; /* expected-error {{use of 'dr251_fred' with tag type that does not match previous declaration}} */
}

#if __STDC_VERSION__ < 202000L
/* WG14 DR252: yes
 * Incomplete argument types when calling non-prototyped functions
 */
void dr252_no_proto();  /* expected-warning {{a function declaration without a prototype is deprecated in all versions of C}} */
void dr252_proto(void); /* expected-note {{'dr252_proto' declared here}} */
void dr252(void) {
  /* It's a constraint violation to pass an argument to a function with a
   * prototype that specifies a void parameter.
   */
  dr252_proto(dr252_no_proto()); /* expected-error {{too many arguments to function call, expected 0, have 1}} */

  /* It's technically UB to pass an incomplete type to a function without a
   * prototype, but Clang treats it as an error.
   */
  dr252_no_proto(dr252_proto()); /* expected-error {{argument type 'void' is incomplete}}
                                    expected-warning {{passing arguments to 'dr252_no_proto' without a prototype is deprecated in all versions of C and is not supported in C2x}}
                                  */
}
#endif /* __STDC_VERSION__ < 202000L */

/* WG14 DR258: yes
 * Ordering of "defined" and macro replacement
 */
void dr258(void) {
  /* We get the diagnostic twice because the argument is used twice in the
   * expansion. */
#define repeat(x) x && x
#if repeat(defined fred) /* expected-warning 2 {{macro expansion producing 'defined' has undefined behavior}} */
#endif

  /* We get no diagnostic because the argument is unused. */
#define forget(x) 0
#if forget(defined fred)
#endif

#undef repeat
#undef forget
}

/* WG14 DR261: yes
 * Constant expressions
 */
void dr261(void) {
  /* This is still an integer constant expression despite the overflow. */
  enum e1 {
    ex1 = __INT_MAX__ + 1  /* expected-warning {{overflow in expression; result is -2147483648 with type 'int'}} */
  };

  /* This is not an integer constant expression, because of the comma operator,
   * but we fold it as a constant expression anyway as a GNU extension.
   */
  enum e2 {
    ex2 = __INT_MAX__ + (0, 1) /* expected-warning {{expression is not an integer constant expression; folding it to a constant is a GNU extension}}
                                  expected-note {{value 2147483648 is outside the range of representable values of type 'int'}}
                                  expected-warning {{left operand of comma operator has no effect}}
                                */
  };

  /* It's a bit weird that we issue a "congratulations, you did the thing"
   * diagnostic, but the diagnostic does help demonstrate that we correctly
   * treat it as a null pointer constant value.
   */
  char *p1 = (1 - 1); /* expected-warning {{expression which evaluates to zero treated as a null pointer constant of type 'char *'}} */

  /* This is an invalid initialization/assignment because the right-hand side
   * does not have pointer to void or pointer to char type and is not the null
   * pointer constant. */
  char *p2 = (42, 1 - 1); /* expected-warning {{incompatible integer to pointer conversion initializing 'char *' with an expression of type 'int'}}
                             expected-warning {{left operand of comma operator has no effect}}
                           */
  p1 = (42, 1 - 1);       /* expected-warning {{incompatible integer to pointer conversion assigning to 'char *' from 'int'}}
                             expected-warning {{left operand of comma operator has no effect}}
                           */

  /* These are both valid. The initialization doesn't require an integer
   * constant expression, nor does the assignment.
   */
  short s1 = 42 + (0, 1); /* c89only-warning {{mixing declarations and code is a C99 extension}}
                             expected-warning {{left operand of comma operator has no effect}}
                           */
  s1 = (42, 69); /* expected-warning {{left operand of comma operator has no effect}} */

  /* These are both valid because they are constant expressions and the value
   * is the null pointer constant.
   */
  p2 = 0;
  p2 = 1 - 1; /* expected-warning {{expression which evaluates to zero treated as a null pointer constant of type 'char *'}} */
}

/* WG14 DR262: yes
 * Maximum size of bit fields
 */
void dr262(void) {
  _Static_assert(sizeof(short) == 2, "short is not two chars?");
  struct S {
    short field : __CHAR_BIT__ * 2; /* ok */
    short other_field : __CHAR_BIT__ * 2 + 1; /* expected-error-re {{width of bit-field 'other_field' ({{[0-9]+}} bits) exceeds the width of its type ({{[0-9]+}} bits)}} */
  };
}

/* WG14 DR263: yes
 * All-zero bits representations
 *
 * This tests that the integer value 0 is not comprised of any non-zero bits,
 * which demonstrates that a value with all zero bits will be treated as the
 * integer value zero.
 */
_Static_assert(__builtin_popcount(0) < 1, "zero is not all zero bits");


/* WG14 DR265: yes
 * Preprocessor arithmetic
 */
#if __UINT_MAX__ == 0xFFFFFFFF
/* Ensure that the literal is interpreted as intptr_t instead of uintptr_t,
 * despite that being the phase 7 behavior being that the literal is unsigned.
 */
#if -0xFFFFFFFF >= 0
#error "Interpreting the literal incorrectly in the preprocessor"
#endif
#endif /* __UINT_MAX__ == 0xFFFFFFFF */


/* WG14 DR266: yes
 * Overflow of sizeof
 */
void dr266(void) {
  /* Some targets support a maximum size which cannot be represented by an
   * unsigned long, and so unsigned long long is used instead. However, C89
   * doesn't have the long long type, so we issue a pedantic warning about it.
   * Disable the warning momentarily so we don't have to add target triples to
   * the RUN lines pinning the targets down concretely.
   */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlong-long"
  (void)sizeof(int[__SIZE_MAX__ / 2][__SIZE_MAX__ / 2]); /* expected-error-re 2 {{array is too large ({{[0-9]+}} elements)}} */
#pragma clang diagnostic pop
}

/* WG14 DR272: yes
 * Type category
 */
void dr272(void) {
  /* The crux of this DR is to confirm that lvalue conversion of the rhs on an
   * assignment expression strips top-level qualifiers, and not all qualifiers,
   * from the resulting expression type.
   */
  const int * volatile ptr;
  (void)_Generic(ptr = 0, const int * : 1); /* expected-warning {{expression with side effects has no effect in an unevaluated context}} */
}

/* WG14 DR277: no
 * Declarations within iteration statements
 */
void dr277(void) {
  /* FIXME: it's a bit silly to issue both of these warnings at the same time
   * in pedantic mode given that they're both effectively the same root cause.
   *
   * C99 6.8.5p3: The declaration part of a for statement shall only declare
   * identifiers for objects having storage class auto or register.
   *
   * FIXME: we don't issue a pedantic warning below for the declaration of E,
   * and its enumerators, none of which declare an object with auto or register
   * storage classes.
   */
  for (enum E { one, two } i = one; i < two; ++i) /* c89only-warning {{variable declaration in for loop is a C99-specific feature}}
                                                     c89only-warning {{GCC does not allow variable declarations in for loop initializers before C99}}
                                                   */
    ;
}

#if __STDC_VERSION__ >= 199901L
/* WG14 DR289: yes
 * Function prototype with [restrict]
 *
 * Ensure that we support [restrict] array syntax as an abstract declarator and
 * not just as a direct declarator.
 */
void dr289(int * restrict const [restrict]);
#endif /* __STDC_VERSION__ >= 199901L */

/* WG14 DR295: yes
 * Incomplete types for function parameters
 */
struct NotCompleted;                    /* expected-note {{forward declaration of 'struct NotCompleted'}} */
void dr295_1(struct NotCompleted);
void dr295_1(struct NotCompleted Val) { /* expected-error {{variable has incomplete type 'struct NotCompleted'}} */
}

/* There's no reason to reject this code, but it's technically undefined
 * behavior, so diagnosing it is reasonable.
 *
 * FIXME: either downgrade this error into a warning or remove it entirely; it
 * doesn't add a whole lot of value as an error.
 */
void dr295_2(void param); /* expected-error {{argument may not have 'void' type}} */

/* WG14 DR298: partial
 * Validity of constant in unsigned long long range
 *
 * I'm giving this one a partial because we fail to pedantically diagnose the
 * use of 'long long' through a constant value. We correctly warn about the
 * type when spelled out and when using an explicit suffix, but we fail to warn
 * otherwise.
 */
#if __LLONG_WIDTH__ >= 64 && __LONG_WIDTH__ < 64
/* This test requires that long long be at least 64-bits and long be smaller
 * because the test is whether the integer literal which is too large to fit in
 * a constant of type long long. This is undefined behavior in C, which means
 * we're free to pick a different type so long as we diagnose the extension
 * appropriately.
 */
void dr298(void) {
  /* FIXME: These uses of the constants need a pedantic warning in C89 mode;
   * we've picked a type that does not exist in C89.
   */
  (void)_Generic(9223372036854775808,     /* expected-warning {{integer literal is too large to be represented in a signed integer type, interpreting as unsigned}} */
                 unsigned long long : 1); /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */
  (void)_Generic(9223372036854775807,
                 long long : 1);          /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */
}
#endif /* __LLONG_WIDTH__ == 64 && __LONG_WIDTH__ < 64 */
