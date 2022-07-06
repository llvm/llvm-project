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
