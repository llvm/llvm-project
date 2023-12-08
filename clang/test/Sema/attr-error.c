// RUN: %clang_cc1 -fsyntax-only -verify %s
#if !__has_attribute(error)
#error "error attribute missing"
#endif

__attribute__((error("don't call me!"))) int good0(void);

__attribute__((error)) // expected-error {{'error' attribute takes one argument}}
int
bad0(void);

int bad1(__attribute__((error("bad1"))) int param); // expected-error {{'error' attribute only applies to functions}}

int bad2(void) {
  __attribute__((error("bad2"))); // expected-error {{'error' attribute cannot be applied to a statement}}
}

__attribute__((error(3))) // expected-error {{expected string literal as argument of 'error' attribute}}
int
bad3(void);

__attribute__((error("foo"), error("foo"))) int good1(void);
__attribute__((error("foo"))) int good1(void);
__attribute__((error("foo"))) int good1(void) {}

__attribute__((error("foo"), warning("foo"))) // expected-error {{'warning' and 'error' attributes are not compatible}}
int
bad4(void);
// expected-note@-3 {{conflicting attribute is here}}

__attribute__((error("foo"))) int bad5(void);   // expected-note {{conflicting attribute is here}}
__attribute__((warning("foo"))) int bad5(void); // expected-error {{'error' and 'warning' attributes are not compatible}}

/*
 * Note: we differ from GCC here; rather than support redeclarations that add
 * or remove this fn attr, we diagnose such differences.
 */

void foo(void);                                     // expected-note {{previous declaration is here}}
__attribute__((error("oh no foo"))) void foo(void); // expected-error {{'error' attribute does not appear on the first declaration}}
