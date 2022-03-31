// RUN: %clang -target x86_64-apple-darwin -mios-simulator-version-min=7 -fsyntax-only %s -Xclang -verify
// RUN: %clang -target x86_64-apple-darwin -arch arm64 -target x86_64-apple-darwin -mios-version-min=7 -fsyntax-only %s -Xclang -verify
// RUN: %clang -target armv7k-apple-watchos -arch armv7k -target armv7k-apple-watchos -fsyntax-only %s -Xclang -verify

// For 64-bit iOS, automatically promote -Wimplicit-function-declaration
// to an error.

void radar_10894044(void) {
  printf("Hi\n"); // expected-error {{implicitly declaring library function 'printf' with type 'int (const char *, ...)'}} expected-note {{include the header <stdio.h> or explicitly provide a declaration for 'printf'}}
  radar_10894044_not_declared(); // expected-error {{implicit declaration of function 'radar_10894044_not_declared' is invalid in C99}}
}
