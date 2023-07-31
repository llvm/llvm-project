// RUN: %clang_cc1 %s -fsyntax-only -std=c99 -pedantic -verify=expected,ext -Wundef -DTRIGRAPHS=1
// RUN: %clang_cc1 %s -fsyntax-only -std=c2x -pedantic -verify=expected,ext -Wundef -DTRIGRAPHS=1
// RUN: %clang_cc1 %s -fsyntax-only -x c++ -pedantic -verify=expected,ext -Wundef -fno-trigraphs
// RUN: %clang_cc1 %s -fsyntax-only -x c++ -std=c++23 -pedantic -ftrigraphs -DTRIGRAPHS=1 -verify=expected,cxx23 -Wundef -Wpre-c++23-compat
// RUN: %clang_cc1 %s -fsyntax-only -x c++ -pedantic -verify=expected,ext -Wundef -ftrigraphs -DTRIGRAPHS=1
// RUN: not %clang_cc1 %s -fsyntax-only -std=c99 -pedantic -Wundef 2>&1 | FileCheck -strict-whitespace %s

#define \u00FC
#define a\u00FD() 0
#ifndef \u00FC
#error "This should never happen"
#endif

#if a\u00FD()
#error "This should never happen"
#endif

#if a\U000000FD()
#error "This should never happen"
#endif

#if a\u{FD}() // ext-warning {{extension}} cxx23-warning {{before C++23}}
#error "This should never happen"
#endif

#if \uarecool // expected-warning{{incomplete universal character name; treating as '\' followed by identifier}} expected-error {{invalid token at start of a preprocessor expression}}
#endif
#if \uwerecool // expected-warning{{\u used with no following hex digits; treating as '\' followed by identifier}} expected-error {{invalid token at start of a preprocessor expression}}
#endif
#if \U0001000  // expected-warning{{incomplete universal character name; treating as '\' followed by identifier}} expected-error {{invalid token at start of a preprocessor expression}}
#endif

// Make sure we reject disallowed UCNs
#define \ufffe // expected-error {{macro name must be an identifier}}
#define \U10000000      // expected-error {{macro name must be an identifier}}
#define \u0061          // expected-error {{character 'a' cannot be specified by a universal character name}} expected-error {{macro name must be an identifier}}
#define \u{fffe}        // expected-error {{macro name must be an identifier}} \
                        // ext-warning {{extension}} cxx23-warning {{before C++23}}
#define \N{ALERT}       // expected-error {{universal character name refers to a control character}} \
                   // expected-error {{macro name must be an identifier}} \
                   // ext-warning {{extension}} cxx23-warning {{before C++23}}
#define \N{WASTEBASKET} // expected-error {{macro name must be an identifier}} \
                        // ext-warning {{extension}} cxx23-warning {{before C++23}}
#define a\u0024a  // expected-error {{character '$' cannot be specified by a universal character name}} \
                  // expected-warning {{requires whitespace after the macro name}}

#if \u0110 // expected-warning {{is not defined, evaluates to 0}}
#endif


#define \u0110 1 / 0
#if \u0110 // expected-error {{division by zero in preprocessor expression}}
#endif

#define STRINGIZE(X) # X

extern int check_size[sizeof(STRINGIZE(\u0112)) == 3 ? 1 : -1];

// Check that we still diagnose disallowed UCNs in #if 0 blocks.
// C99 5.1.1.2p1 and C++11 [lex.phases]p1 dictate that preprocessor tokens are
// formed before directives are parsed.
// expected-error@+4 {{character 'a' cannot be specified by a universal character name}}
#if 0
#define \ufffe // okay
#define \U10000000 // okay
#define \u0061 // error, but -verify only looks at comments outside #if 0
#endif


// A UCN formed by token pasting is undefined in both C99 and C++.
// Right now we don't do anything special, which causes us to coincidentally
// accept the first case below but reject the second two.
#define PASTE(A, B) A ## B
extern int PASTE(\, u00FD);
extern int PASTE(\u, 00FD); // expected-warning{{\u used with no following hex digits}}
extern int PASTE(\u0, 0FD); // expected-warning{{incomplete universal character name}}
#ifdef __cplusplus
// expected-error@-3 {{expected unqualified-id}}
// expected-error@-3 {{expected unqualified-id}}
#else
// expected-error@-6 {{expected identifier}}
// expected-error@-6 {{expected identifier}}
#endif


// A UCN produced by line splicing is valid in C99 but undefined in C++.
// Since undefined behavior can do anything including working as intended,
// we just accept it in C++ as well.;
#define newline_1_\u00F\
C 1
#define newline_2_\u00\
F\
C 1
#define newline_3_\u\
00\
FC 1
#define newline_4_\\
u00FC 1
#define newline_5_\\
u\
\
0\
0\
F\
C 1

#if (newline_1_\u00FC && newline_2_\u00FC && newline_3_\u00FC && \
     newline_4_\u00FC && newline_5_\u00FC)
#else
#error "Line splicing failed to produce UCNs"
#endif


#define capital_u_\U00FC
// expected-warning@-1 {{incomplete universal character name}} expected-note@-1 {{did you mean to use '\u'?}} expected-warning@-1 {{whitespace}}
// CHECK: note: did you mean to use '\u'?
// CHECK-NEXT: {{^  .* | #define capital_u_\U00FC}}
// CHECK-NEXT: {{^      |                    \^}}
// CHECK-NEXT: {{^      |                    u}}

#define \u{}           // expected-warning {{empty delimited universal character name; treating as '\' 'u' '{' '}'}} expected-error {{macro name must be an identifier}}
#define \u1{123}       // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}} expected-error {{macro name must be an identifier}}
#define \u{123456789}  // expected-error {{hex escape sequence out of range}} expected-error {{macro name must be an identifier}}
#define \u{            // expected-warning {{incomplete delimited universal character name; treating as '\' 'u' '{' identifier}} expected-error {{macro name must be an identifier}}
#define \u{fgh}        // expected-warning {{incomplete delimited universal character name; treating as '\' 'u' '{' identifier}} expected-error {{macro name must be an identifier}}
#define \N{
// expected-warning@-1 {{incomplete delimited universal character name; treating as '\' 'N' '{' identifier}}
// expected-error@-2 {{macro name must be an identifier}}
#define \N{}           // expected-warning {{empty delimited universal character name; treating as '\' 'N' '{' '}'}} expected-error {{macro name must be an identifier}}
#define \N{NOTATHING}  // expected-error {{'NOTATHING' is not a valid Unicode character name}} \
                       // expected-error {{macro name must be an identifier}}
#define \NN            // expected-warning {{incomplete universal character name; treating as '\' followed by identifier}} expected-error {{macro name must be an identifier}}
#define \N{GREEK_SMALL-LETTERALPHA}  // expected-error {{'GREEK_SMALL-LETTERALPHA' is not a valid Unicode character name}} \
                                     // expected-note {{characters names in Unicode escape sequences are sensitive to case and whitespaces}}
#define \N{🤡}  // expected-error {{'🤡' is not a valid Unicode character name}} \
                // expected-error {{macro name must be an identifier}}

#define CONCAT(A, B) A##B
int CONCAT(\N{GREEK
, CAPITALLETTERALPHA});
// expected-error@-2 {{expected}} \
// expected-warning@-2 {{incomplete delimited universal character name}}

int \N{\
LATIN CAPITAL LETTER A WITH GRAVE};
//ext-warning@-2 {{extension}} cxx23-warning@-2 {{before C++23}}

#ifdef TRIGRAPHS
int \N??<GREEK CAPITAL LETTER ALPHA??> = 0; // cxx23-warning {{before C++23}} \
                                            //ext-warning {{extension}}\
                                            // expected-warning 2{{trigraph converted}}

int a\N{LATIN CAPITAL LETTER A WITH GRAVE??>; // expected-warning {{trigraph converted}}
#endif

#ifndef TRIGRAPHS
int a\N{LATIN CAPITAL LETTER A WITH GRAVE??>;
// expected-warning@-1 {{trigraph ignored}}\
// expected-warning@-1 {{incomplete}}\
// expected-error@-1 {{expected unqualified-id}}
#endif

// GH64161
int A\N{LEFT-TO-RIGHT OVERRIDE}; // expected-error {{character <U+202D> not allowed in an identifier}}
