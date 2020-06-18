// RUN: not %clang_cc1 -fsyntax-only -fmessage-length=75 -o /dev/null -x c < %s 2>&1 | FileCheck %s -strict-whitespace
// REQUIRES: utf8-capable-terminal

// Test case for the text diagnostics source column conversion crash.

// This test case tries to check the error diagnostic message printer, which is
// responsible to create the code snippet shorter than the message-length (in
// number of columns.)
//
// The error diagnostic message printer should be able to handle the non-ascii
// characters without any segmentation fault or assertion failure.  If your
// changes to clang frontend crashes this case, it is likely that you are mixing
// column index with byte index which are two totally different concepts.

// NOTE: This file is encoded in UTF-8 and intentionally contains some
// non-ASCII characters.

__attribute__((format(printf, 1, 2)))
extern int printf(const char *fmt, ...);

void test1(Unknown* b);  // αααα αααα αααα αααα αααα αααα αααα αααα αααα αααα αααα
// CHECK: unknown type name 'Unknown'
// CHECK-NEXT: void test1(Unknown* b);  // αααα αααα αααα αααα αααα αααα αααα ααα...
// CHECK-NEXT: {{^           \^$}}

void test2(Unknown* b);  // αααα αααα αααα αααα αααα αααα αααα αααα αααα

// CHECK: unknown type name 'Unknown'
// CHECK-NEXT: void test2(Unknown* b);  // αααα αααα αααα αααα αααα αααα αααα αααα αααα
// CHECK-NEXT: {{^           \^$}}

void test3() {
   /* αααα αααα αααα αααα αααα αααα αααα αααα αααα αααα */ printf("%d", "s");
}
// CHECK:       format specifies type 'int' but the argument has
// CHECK:       type 'char *'
// CHECK-NEXT:   ...αααα αααα αααα αααα αααα αααα αααα αααα αααα */ printf("%d", "s");
// CHECK-NEXT: {{^                                                             ~~   \^~~$}}
// CHECK-NEXT: {{^                                                             %s$}}
