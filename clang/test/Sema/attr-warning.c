// RUN: %clang_cc1 -fsyntax-only -verify %s
#if !__has_attribute(warning)
#warning "warning attribute missing"
#endif

__attribute__((warning("don't call me!"))) int good0(void);

__attribute__((warning)) // expected-error {{'warning' attribute takes one argument}}
int
bad0(void);

int bad1(__attribute__((warning("bad1"))) int param); // expected-error {{'warning' attribute only applies to functions}}

int bad2(void) {
  __attribute__((warning("bad2"))); // expected-error {{'warning' attribute cannot be applied to a statement}}
}

__attribute__((warning(3))) // expected-error {{'warning' attribute requires a string}}
int
bad3(void);

__attribute__((warning("foo"), warning("foo"))) int good1(void);
__attribute__((warning("foo"))) int good1(void);
__attribute__((warning("foo"))) int good1(void) {}

__attribute__((warning("foo"), error("foo"))) // expected-error {{'error' and 'warning' attributes are not compatible}}
int
bad4(void);
// expected-note@-3 {{conflicting attribute is here}}

/*
 * Note: we differ from GCC here; rather than support redeclarations that add
 * or remove this fn attr, we diagnose such differences.
 */

void foo(void);                                       // expected-note {{previous declaration is here}}
__attribute__((warning("oh no foo"))) void foo(void); // expected-error {{'warning' attribute does not appear on the first declaration}}
