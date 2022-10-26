/* RUN: %clang_cc1 -std=c89 -fsyntax-only -Wvla -verify=expected,c89only -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -fsyntax-only -Wvla -verify=expected,c99andup -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -fsyntax-only -Wvla -verify=expected,c99andup -pedantic %s
   RUN: %clang_cc1 -std=c17 -fsyntax-only -Wvla -verify=expected,c99andup -pedantic %s
   RUN: %clang_cc1 -std=c2x -fsyntax-only -Wvla -verify=expected,c99andup -pedantic %s
 */

/* The following are DRs which do not require tests to demonstrate
 * conformance or nonconformance.
 *
 * WG14 DR300: yes
 * Translation-time expresssion evaluation
 *
 * WG14 DR301: yes
 * Meaning of FE_* macros in <fenv.h>
 *
 * WG14 DR303: yes
 * Breaking up the very long sentence describing preprocessing directive
 *
 * WG14 DR307: yes
 * Clarifiying arguments vs. parameters
 *
 * WG14 DR308: yes
 * Clarify that source files et al. need not be "files"
 *
 * WG14 DR310: yes
 * Add non-corner case example of trigraphs
 *
 * WG14 DR312: yes
 * Meaning of "known constant size"
 *
 * WG14 DR333: yes
 * Missing Predefined Macro Name
 *
 * WG14 DR342: dup 340
 * VLAs and conditional expressions
 */


/* WG14 DR302: yes
 * Adding underscore to portable include file name character set
 */
#include "./abc_123.h"
#ifndef WE_SUPPORT_DR302
#error "Oh no, we don't support DR302 after all!"
#endif

/* WG14 DR304: yes
 * Clarifying illegal tokens in #if directives
 */
/* expected-error@+3 {{invalid token at start of a preprocessor expression}}
   expected-warning@+3 {{missing terminating ' character}}
 */
#if 'test
#endif

/* WG14 DR305: yes
 * Clarifying handling of keywords in #if directives
 */
#if int
#error "We definitely should not have gotten here"
#endif

/* WG14 DR306: yes
 * Clarifying that rescanning applies to object-like macros
 */
#define REPLACE 1
#define THIS REPLACE
#if THIS != 1
#error "We definitely should not have gotten here"
#endif

/* WG14 DR309: yes
 * Clarifying trigraph substitution
 */
int dr309??(1??) = { 1 }; /* expected-warning {{trigraph converted to '[' character}}
                             expected-warning {{trigraph converted to ']' character}}
                           */

/* WG14 DR311: yes
 * Definition of variably modified types
 */
void dr311(int x) {
  typedef int vla[x]; /* expected-warning {{variable length array}} */

  /* Ensure that a constant array of variable-length arrays are still
   * considered a variable-length array.
   */
  vla y[3]; /* expected-warning {{variable length array}} */
}

/* WG14 DR313: yes
 * Incomplete arrays of VLAs
 */
void dr313(int i) {
  int c[][i] = { 0 }; /* expected-error {{variable-sized object may not be initialized}}
                         expected-warning {{variable length array}}
                       */
}

/* WG14 DR315: yes
 * Implementation-defined bit-field types
 */
struct dr315_t {
  unsigned long long a : 37; /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */
  unsigned long long b : 37; /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */

  short c : 8;
  short d : 8;
} dr315;
_Static_assert(sizeof(dr315.a + dr315.b) == sizeof(unsigned long long), ""); /* c89only-warning {{'long long' is an extension when C99 mode is not enabled}} */
/* Demonstrate that integer promotions still happen when less than the width of
 * an int.
 */
_Static_assert(sizeof(dr315.c + dr315.d) == sizeof(int), "");

#if __STDC_VERSION__ < 202000L
/* WG14 DR316: yes
 * Unprototyped function types
 */
void dr316_1(a) int a; {}  /* expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C2x}} */
void (*dr316_1_ptr)(int, int, int) = dr316_1;

/* WG14 DR317: yes
 * Function definitions with empty parentheses
 *
 * Despite the function with empty parens being a definition, this does not
 * provide a prototype for the function. However, calling the function with
 * arguments is undefined behavior, so it is defensible for us to warn the user
 * about it. They key point to this DR is that we give the "without a
 * prototype" warnings to demonstrate we don't give this function a prototype.
 */
void dr317_1() {}  /* expected-warning {{a function declaration without a prototype is deprecated in all versions of C}} */
void dr317_2(void) {
  if (0)
    dr317_1(1); /* expected-warning {{too many arguments in call to 'dr317_1'}}
                   expected-warning {{passing arguments to 'dr317_1' without a prototype is deprecated in all versions of C and is not supported in C2x}}
                 */
}
#endif /* __STDC_VERSION__ < 202000L */

/* WG14 DR320: yes
 * Scope of variably modified type
 */
int dr320_v;
typedef int dr320_t[dr320_v]; /* c89only-warning {{variable length arrays are a C99 feature}}
                                 expected-error {{variable length array declaration not allowed at file scope}}
                                 c99andup-warning {{variable length array used}}
                               */
void dr320(int okay[dr320_v]) { /* c89only-warning {{variable length arrays are a C99 feature}}
                                   c99andup-warning {{variable length array used}}
                                 */
  typedef int type[dr320_v]; /* c89only-warning {{variable length arrays are a C99 feature}}
                                c99andup-warning {{variable length array used}}
                              */
  extern type bad;  /* expected-error {{variable length array declaration cannot have 'extern' linkage}} */

  /* C99 6.7.5.2p2, second sentence. */
  static type fine; /* expected-error {{variable length array declaration cannot have 'static' storage duration}} */
}

/* WG14 DR321: yes
 * Wide character code values for members of the basic character set
 */
#define DR321 (\
    ' ' == L' ' && '\t' == L'\t' && '\v' == L'\v' && '\r' == L'\r' &&           \
    '\n' == L'\n' &&                                                            \
    'a' == L'a' && 'b' == L'b' && 'c' == L'c' && 'd' == L'd' && 'e' == L'e' &&  \
    'f' == L'f' && 'g' == L'g' && 'h' == L'h' && 'i' == L'i' && 'j' == L'j' &&  \
    'k' == L'k' && 'l' == L'l' && 'm' == L'm' && 'n' == L'n' && 'o' == L'o' &&  \
    'p' == L'p' && 'q' == L'q' && 'r' == L'r' && 's' == L's' && 't' == L't' &&  \
    'u' == L'u' && 'v' == L'v' && 'w' == L'w' && 'x' == L'x' && 'y' == L'y' &&  \
    'z' == L'z' &&                                                              \
    'A' == L'A' && 'B' == L'B' && 'C' == L'C' && 'D' == L'D' && 'E' == L'E' &&  \
    'F' == L'F' && 'G' == L'G' && 'H' == L'H' && 'I' == L'I' && 'J' == L'J' &&  \
    'K' == L'K' && 'L' == L'L' && 'M' == L'M' && 'N' == L'N' && 'O' == L'O' &&  \
    'P' == L'P' && 'Q' == L'Q' && 'R' == L'R' && 'S' == L'S' && 'T' == L'T' &&  \
    'U' == L'U' && 'V' == L'V' && 'W' == L'W' && 'X' == L'X' && 'Y' == L'Y' &&  \
    'Z' == L'Z' &&                                                              \
    '0' == L'0' && '1' == L'1' && '2' == L'2' && '3' == L'3' && '4' == L'4' &&  \
    '5' == L'5' && '6' == L'6' && '7' == L'7' && '8' == L'8' &&                 \
    '9' == L'9' &&                                                              \
    '_' == L'_' && '{' == L'{' && '}' == L'}' && '[' == L'[' && ']' == L']' &&  \
    '#' == L'#' && '(' == L'(' && ')' == L')' && '<' == L'<' && '>' == L'>' &&  \
    '%' == L'%' && ':' == L':' && ';' == L';' && '.' == L'.' && '?' == L'?' &&  \
    '*' == L'*' && '+' == L'+' && '-' == L'-' && '/' == L'/' && '^' == L'^' &&  \
    '&' == L'&' && '|' == L'|' && '~' == L'~' && '!' == L'!' && '=' == L'=' &&  \
    ',' == L',' && '\\' == L'\\' && '"' == L'"' && '\'' == L'\''                \
  )
#if __STDC_MB_MIGHT_NEQ_WC__
#ifndef __FreeBSD__ // PR22208, FreeBSD expects us to give a bad (but conforming) answer here.
_Static_assert(!DR321, "__STDC_MB_MIGHT_NEQ_WC__ but all basic source characters have same representation");
#endif
#else
_Static_assert(DR321, "!__STDC_MB_MIGHT_NEQ_WC__ but some character differs");
#endif

/* WG14 DR328: partial
 * String literals in compound literal initialization
 *
 * DR328 is implemented properly in terms of allowing string literals, but is
 * not implemented. See DR339 (marked as a duplicate of this one) for details.
 */
const char *dr328_v = (const char *){"this is a string literal"}; /* c89only-warning {{compound literals are a C99-specific feature}} */
void dr328(void) {
  const char *val = (const char *){"also a string literal"}; /* c89only-warning {{compound literals are a C99-specific feature}} */
}

/* WG14 DR335: yes
 * _Bool bit-fields
 *
 * See dr335.c also, which tests the runtime behavior of the part of the DR
 * which will compile.
 */
void dr335(void) {
  struct bits_ {
    _Bool bbf3 : 3; /* expected-error {{width of bit-field 'bbf3' (3 bits) exceeds the width of its type (1 bit)}}
                       c89only-warning {{'_Bool' is a C99 extension}}
                     */
  };
}

/* WG14 DR339: dup 328
 * Variably modified compound literals
 *
 * This DR is marked as a duplicate of DR328, see that DR for further
 * details.
 *
 * FIXME: we should be diagnosing this compound literal as creating a variably-
 * modified type at file scope, as we would do for a file scope variable.
 */
extern int dr339_v;
void *dr339 = &(int (*)[dr339_v]){ 0 }; /* c89only-warning {{variable length arrays are a C99 feature}}
                                           c99andup-warning {{variable length array used}}
                                           c89only-warning {{compound literals are a C99-specific feature}}
                                         */

/* WG14 DR340: yes
 * Composite types for variable-length arrays
 *
 * The DR made this behavior undefined because implementations disagreed on the
 * behavior. For this DR, Clang accepts the code and GCC rejects it. It's
 * unclear whether the Clang behavior is intentional, but because the code is
 * UB, any behavior is acceptable.
 */
#if __STDC_VERSION__ < 202000L
void dr340(int x, int y) {
  typedef void (*T1)(int);
  typedef void (*T2)(); /* expected-warning {{a function declaration without a prototype is deprecated in all versions of C}} */

  T1 (*a)[] = 0;
  T2 (*b)[x] = 0;       /* c89only-warning {{variable length arrays are a C99 feature}}
                           c99andup-warning {{variable length array used}}
                         */
  (y ? a : b)[0][0]();
}
#endif /* __STDC_VERSION__ < 202000L */

/* WG14 DR341: yes
 * [*] in abstract declarators
 */
void dr341_1(int (*)[*]);                  /* c89only-warning {{variable length arrays are a C99 feature}}
                                              c99andup-warning {{variable length array used}}
                                            */
void dr341_2(int (*)[sizeof(int (*)[*])]); /* expected-error {{star modifier used outside of function prototype}} */

/* WG14 DR343: yes
 * Initializing qualified wchar_t arrays
 */
void dr343(void) {
  const __WCHAR_TYPE__ x[] = L"foo";
}

/* WG14 DR344: yes
 * Casts in preprocessor conditional expressions
 *
 * Note: this DR removed a constraint about not containing casts because there
 * are no keywords, therefore no types to cast to, so casts simply don't exist
 * as a construct during preprocessing.
 */
#if (int)+0
#error "this should not be an error, we shouldn't get here"
#else
/* expected-error@+1 {{"reached"}} */
#error "reached"
#endif

/* WG14 DR345: yes
 * Where does parameter scope start?
 */
void f(long double f,
       char (**a)[10 * sizeof f]) {
  _Static_assert(sizeof **a == sizeof(long double) * 10, "");
}
