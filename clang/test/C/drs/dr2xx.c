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
   (void)_Generic(s + sl, __typeof__(s) : 1, unsigned long : 1, unsigned int : 1);
   (void)_Generic(p + sl, __typeof__(p) : 1, signed long : 1, signed int : 1);
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
