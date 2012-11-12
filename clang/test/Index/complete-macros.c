// Note: the run lines follow their respective tests, since line/column
// matter in this test.
#define FOO(Arg1,Arg2) foobar
#define nil 0
#undef FOO
void f() {

}

void g(int);

void f2() {
  int *ip = nil;
  ip = nil;
  g(nil);
}

#define variadic1(...)
#define variadic2(args...)
#define variadic3(args, ...)
#define variadic4(first, second, args, ...)
#define variadic5(first, second, args ...)

void test_variadic() {
  
}

// RUN: c-index-test -code-completion-at=%s:7:1 %s | FileCheck -check-prefix=CHECK-CC0 %s
// CHECK-CC0-NOT: FOO
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:7:1 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: macro definition:{TypedText FOO}{LeftParen (}{Placeholder Arg1}{Comma , }{Placeholder Arg2}{RightParen )}
// RUN: c-index-test -code-completion-at=%s:13:13 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:14:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:14:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: macro definition:{TypedText nil} (32)
// RUN: c-index-test -code-completion-at=%s:15:5 %s | FileCheck -check-prefix=CHECK-CC3 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:15:5 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: macro definition:{TypedText nil} (65)
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:25:2 %s | FileCheck -check-prefix=CHECK-VARIADIC %s
// CHECK-VARIADIC: macro definition:{TypedText variadic1}{LeftParen (}{Placeholder ...}{RightParen )} (70)
// CHECK-VARIADIC: macro definition:{TypedText variadic2}{LeftParen (}{Placeholder args...}{RightParen )} (70)
// CHECK-VARIADIC: macro definition:{TypedText variadic3}{LeftParen (}{Placeholder args, ...}{RightParen )} (70)
// CHECK-VARIADIC: macro definition:{TypedText variadic4}{LeftParen (}{Placeholder first}{Comma , }{Placeholder second}{Comma , }{Placeholder args, ...}{RightParen )} (70)
// CHECK-VARIADIC: macro definition:{TypedText variadic5}{LeftParen (}{Placeholder first}{Comma , }{Placeholder second}{Comma , }{Placeholder args...}{RightParen )} (70)
