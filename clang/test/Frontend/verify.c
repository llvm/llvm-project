// RUN: %clang_cc1 -DTEST1 -verify %s
// RUN: not %clang_cc1 -DTEST2 -verify %s 2>&1 | FileCheck -check-prefix=CHECK2 %s
// RUN: not %clang_cc1 -DTEST3 -verify %s 2>&1 | FileCheck -check-prefix=CHECK3 %s
// RUN: not %clang_cc1 -DTEST4 -verify %s 2>&1 | FileCheck -check-prefix=CHECK4 %s
// RUN: not %clang_cc1 -DTEST5 -verify %s 2>&1 | FileCheck -check-prefix=CHECK5 %s

// expected-warning@ malformed
// expected-error@7 1 {{missing or invalid line number}}

// expected-warning@0 malformed
// expected-error@10 {{missing or invalid line number}}

// expected-warning@-50 malformed
// expected-error@13 {{missing or invalid line number}}

// expected-warning malformed
// expected-error@16 {{cannot find start}}

// expected-error 0+ {{should also be ignored}}

#ifdef TEST1
#if 0
// expected-error {{should be ignored}}
#endif
// eexpected-error {{should also be ignored: unrecognised directive}}
#error should not be ignored
// expected-error@-1 1+ {{should not be ignored}}

#line 90
unexpected a; // expected-error@+0 + {{unknown type}}

#line 60
unexpected b; // expected-error@33 1-1 {{unknown type}}

// expected-error@+2 {{file not found}} check that multi-line handled correctly: \

#include "missing_header_file.include"
#endif

#ifdef TEST2
#define MACRO some_value // expected-error {{define_error}}
#undef MACRO extra_token // expected-warning {{undef_error}}
#line -2                 // expected-error {{line_error}}
#error AAA               // expected-error {{BBB}} <- this shall be part of diagnostic
#warning CCC             // expected-warning {{DDD}} <- this shall be part of diagnostic

#if 0
// This is encapsulated in "#if 0" so that the expected-* checks below
// are not inadvertently included in the diagnostic checking!

//      CHECK2: error: 'expected-error' diagnostics expected but not seen:
// CHECK2-NEXT:   Line 41: define_error
// CHECK2-NEXT:   Line 43: line_error
// CHECK2-NEXT: error: 'expected-error' diagnostics seen but not expected:
// CHECK2-NEXT:   Line 43: #line directive requires a positive integer argument
// CHECK2-NEXT:   Line 44: AAA // expected-error {{[{][{]BBB[}][}]}} <- this shall be part of diagnostic
// CHECK2-NEXT: error: 'expected-warning' diagnostics expected but not seen:
// CHECK2-NEXT:   Line 42: undef_error
// CHECK2-NEXT: error: 'expected-warning' diagnostics seen but not expected:
// CHECK2-NEXT:   Line 42: extra tokens at end of #undef directive
// CHECK2-NEXT:   Line 45: CCC // expected-warning {{[{][{]DDD[}][}]}} <- this shall be part of diagnostic
// CHECK2-NEXT: 7 errors generated.
#endif
#endif

#ifdef TEST3
#ifndef TEST3         // expected-note {{line_67}}
                      // expected-note {{line_68_ignored}}
# ifdef UNDEFINED     // expected-note {{line_69_ignored}}
# endif               // expected-note {{line_70_ignored}}
#elif defined(TEST3)  // expected-note {{line_71}}
# if 1                // expected-note {{line_72}}
                      // expected-note {{line_73}}
# else                // expected-note {{line_74}}
                      // expected-note {{line_75_ignored}}
#  ifndef TEST3       // expected-note {{line_76_ignored}}
#  endif              // expected-note {{line_77_ignored}}
# endif               // expected-note {{line_78}}
#endif

//      CHECK3: error: 'expected-note' diagnostics expected but not seen:
// CHECK3-NEXT:   Line 67: line_67
// CHECK3-NEXT:   Line 71: line_71
// CHECK3-NEXT:   Line 72: line_72
// CHECK3-NEXT:   Line 73: line_73
// CHECK3-NEXT:   Line 74: line_74
// CHECK3-NEXT:   Line 78: line_78
// CHECK3-NEXT: 6 errors generated.
#endif

#ifdef TEST4
#include "missing_header_file.include" // expected-error {{include_error}}

//      CHECK4: error: 'expected-error' diagnostics expected but not seen:
// CHECK4-NEXT:   Line 92: include_error
// CHECK4-NEXT: error: 'expected-error' diagnostics seen but not expected:
// CHECK4-NEXT:   Line 92: 'missing_header_file.include' file not found
// CHECK4-NEXT: 2 errors generated.
#endif

#ifdef TEST5
#include "verify-directive.h"
// expected-error@50 {{source file test}}

//      CHECK5: error: 'expected-error' diagnostics expected but not seen:
// CHECK5-NEXT:   Line 1 (directive at {{.*}}verify-directive.h:2): include file test
// CHECK5-NEXT:   Line 50 (directive at {{.*}}verify.c:103): source file test
// CHECK5-NEXT: 2 errors generated.
#endif

#if 0
// RUN: not %clang_cc1 -verify %t.invalid 2>&1 | FileCheck -check-prefix=CHECK6 %s

//      CHECK6: error: no expected directives found: consider use of 'expected-no-diagnostics'
// CHECK6-NEXT: error: 'expected-error' diagnostics seen but not expected:
// CHECK6-NEXT:   (frontend): error reading '{{.*}}verify.c.tmp.invalid'
// CHECK6-NEXT: 2 errors generated.

// RUN: printf '//expected-error@2{{1}}\n#error 2\n' | not %clang_cc1 -verify 2>&1 | FileCheck -check-prefix=CHECK7 %s

//      CHECK7: error: 'expected-error' diagnostics expected but not seen:
// CHECK7-NEXT:   Line 2 (directive at <stdin>:1): 1
// CHECK7-NEXT: error: 'expected-error' diagnostics seen but not expected:
// CHECK7-NEXT:   Line 2: 2
// CHECK7-NEXT: 2 errors generated.
#endif

#ifdef TEST8
// RUN: not %clang_cc1 -DTEST8 -verify %s 2>&1 | FileCheck -check-prefix=CHECK8 %s

// expected-warning@nonexistent-file:1 {{ }}
// expected-error@-1 {{file 'nonexistent-file' could not be located}}

// expected-warning@verify-directive.h: {{ }}
// expected-error@-1 {{missing or invalid line number}}

// expected-warning@verify-directive.h:0 {{ }}
// expected-error@-1 {{missing or invalid line number}}

// expected-warning@verify-directive.h:0*{{ }}
// expected-error@-1 {{missing or invalid line number}}

// expected-warning@verify-directive.h:*0{{ }}
// syntactically ok -- means match in any line for 0 occurrences.

// expected-warning@verify-directive.h:1 {{diagnostic}}

//      CHECK8: error: 'expected-warning' diagnostics expected but not seen:
// CHECK8-NEXT:   File {{.*}}verify-directive.h Line 1 (directive at {{.*}}verify.c:146): diagnostic
// CHECK8-NEXT: 1 error generated.
#endif


#ifdef TEST9
// RUN: not %clang_cc1 -DTEST9 -verify=what %s 2>&1 | FileCheck -check-prefix=CHECK9 %s

// what-error {{huh?}}
// CHECK9: error: 'what-error' diagnostics expected but not seen:
#endif

#ifdef TEST_WIDE_DELIM
// RUN: not %clang_cc1 -DTEST_WIDE_DELIM -verify %s 2>&1 | FileCheck -check-prefix=CHECK-WIDE-DELIM %s

// expected-error {{{some message with {{}} in it}}}
// expected-error {{{some message with  {}} in it}}}
// expected-error {{{some message with {{}  in it}}}

// expected-error-re {{{some {{.*}} regex with double braces}}}
// expected-error-re {{{some message with {{}  in it}}}

// expected-error {{{mismatched delim}}
// expected-error-re {{{mismatched re {{.*} }}}
// expected-error-re {{{no regex}}}

#if 0
//      CHECK-WIDE-DELIM: error: 'expected-error' diagnostics expected but not seen:
// CHECK-WIDE-DELIM-NEXT:   verify.c Line 164: some message with {{[{]{}[}]}} in it
// CHECK-WIDE-DELIM-NEXT:   verify.c Line 165: some message with  {}} in it
// CHECK-WIDE-DELIM-NEXT:   verify.c Line 166: some message with {{[{]{[}]}}  in it
// CHECK-WIDE-DELIM-NEXT:   verify.c Line 168: {some {{.*}} regex with double braces
// CHECK-WIDE-DELIM-NEXT: error: 'expected-error' diagnostics seen but not expected:
// CHECK-WIDE-DELIM-NEXT:   verify.c Line 169: cannot find end ('}}') of expected regex
// CHECK-WIDE-DELIM-NEXT:   verify.c Line 171: cannot find end ('}}}') of expected string
// CHECK-WIDE-DELIM-NEXT:   verify.c Line 172: cannot find end ('}}') of expected regex
// CHECK-WIDE-DELIM-NEXT:   verify.c Line 173: cannot find start of regex ('{{[{][{]}}') in {no regex
// CHECK-WIDE-DELIM-NEXT: 8 errors generated.
#endif

#endif
