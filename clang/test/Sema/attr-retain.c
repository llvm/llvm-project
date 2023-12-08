// RUN: %clang_cc1 -fsyntax-only -verify %s -Wunused-function

/// We allow 'retain' on non-ELF targets because 'retain' is often used together
/// with 'used'. 'used' has GC root semantics on macOS and Windows. We want
/// users to just write retain,used and don't need to dispatch on binary formats.

extern char test1[] __attribute__((retain));       // expected-warning {{'retain' attribute ignored on a non-definition declaration}}
extern const char test2[] __attribute__((retain)); // expected-warning {{'retain' attribute ignored on a non-definition declaration}}
const char test3[] __attribute__((retain)) = "";

struct __attribute__((retain)) s { // expected-warning {{'retain' attribute only applies to variables with non-local storage, functions, and Objective-C methods}}
};

void foo(void) {
  static int a __attribute__((retain));
  int b __attribute__((retain)); // expected-warning {{'retain' attribute only applies to variables with non-local storage, functions, and Objective-C methods}}
  (void)a;
  (void)b;
}

__attribute__((retain,used)) static void f0(void) {}
__attribute__((retain)) static void f1(void) {} // expected-warning {{unused function 'f1'}}
__attribute__((retain)) void f2(void) {}

/// Test attribute merging.
int tentative;
int tentative __attribute__((retain));
extern int tentative;
int tentative = 0;
