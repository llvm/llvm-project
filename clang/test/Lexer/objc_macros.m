// RUN: %clang_cc1 -fsyntax-only "-triple" "x86_64-apple-macosx10.10.0" -fobjc-runtime-has-weak -fobjc-weak %s -verify %s

#define __strong
// expected-warning@-1 {{ignoring redefinition of Objective-C qualifier macro}}
#define __weak
// expected-warning@-1 {{ignoring redefinition of Objective-C qualifier macro}}
#define __unsafe_unretained
// expected-warning@-1 {{ignoring redefinition of Objective-C qualifier macro}}
#define __autoreleased
// No warning because this is the default expansion anyway.

// Check that this still expands to the right text.
void test(void) {
  goto label; // expected-error {{cannot jump from this goto statement to its label}}
  __weak id x; // expected-note {{jump bypasses initialization of __weak variable}}
label:
  return;
}

#undef __strong
#define __strong
// No warning.
