// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wbool-operation %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -x c -fsyntax-only -Wbitwise-instead-of-logical -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c -fsyntax-only -Wbool-operation -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wbool-operation %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wbitwise-instead-of-logical -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wbool-operation -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#ifdef __cplusplus
typedef bool boolean;
#else
typedef _Bool boolean;
#endif

boolean foo(void);
boolean bar(void);
boolean baz(void) __attribute__((const));
void sink(boolean);

#define FOO foo()

void test(boolean a, boolean b, int *p, volatile int *q, int i) {
  b = a | b;
  b = foo() | a;
  b = (p != 0) | (*p == 42);   // FIXME: also warn for a non-volatile pointer dereference
  b = foo() | (*q == 42);      // expected-warning {{use of bitwise '|' with boolean operands}}
                               // expected-note@-1 {{cast one or both operands to int to silence this warning}}
  b = foo() | (int)(*q == 42); // OK, no warning expected
  b = a | foo();
  b = (int)a | foo();     // OK, no warning expected
  b = foo() | bar();      // expected-warning {{use of bitwise '|' with boolean operands}}
                          // expected-note@-1 {{cast one or both operands to int to silence this warning}}
                          // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:14}:"||"
  b = foo() | !bar();     // expected-warning {{use of bitwise '|' with boolean operands}}
                          // expected-note@-1 {{cast one or both operands to int to silence this warning}}
                          // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:14}:"||"
  b = foo() | (int)bar(); // OK, no warning expected
  b = a | baz();
  b = bar() | FOO;        // expected-warning {{use of bitwise '|' with boolean operands}}
                          // expected-note@-1 {{cast one or both operands to int to silence this warning}}
                          // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:13-[[@LINE-2]]:14}:"||"
  b = foo() | (int)FOO;   // OK, no warning expected
  b = b | foo();
  b = bar() | (i > 4);
  b = (i == 7) | foo();
#ifdef __cplusplus
  b = foo() bitor bar();  // expected-warning {{use of bitwise '|' with boolean operands}}
                          // expected-note@-1 {{cast one or both operands to int to silence this warning}}
#endif

  if (foo() | bar())      // expected-warning {{use of bitwise '|' with boolean operands}}
                          // expected-note@-1 {{cast one or both operands to int to silence this warning}}
    ;

  sink(a | b);
  sink(a | foo());
  sink(foo() | bar());    // expected-warning {{use of bitwise '|' with boolean operands}}
                          // expected-note@-1 {{cast one or both operands to int to silence this warning}}

  int n = i + 10;
  b = (n | (n - 1));
}
