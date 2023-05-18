// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef struct NotAClass {
  int a, b;
} NotAClass;

void foo(void) {
  [NotAClass nonexistent_method]; // expected-error {{receiver type 'NotAClass' (aka 'struct NotAClass') is not an Objective-C class}}
}
