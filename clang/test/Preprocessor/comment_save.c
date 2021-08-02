// RUN: %clang_cc1 -E -C %s | FileCheck -strict-whitespace %s

// foo
// CHECK: // foo

/* bar */
// CHECK: /* bar */

#if FOO
#endif
/* baz */
// CHECK: /* baz */

_Pragma("unknown") // after unknown pragma
// CHECK: #pragma unknown
// CHECK-NEXT: #
// CHECK-NEXT: // after unknown pragma

_Pragma("comment(\"abc\")") // after known pragma
// CHECK: #pragma comment("abc")
// CHECK-NEXT: #
// CHECK-NEXT: // after known pragma
