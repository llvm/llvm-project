// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple thumbv7-eabi -fsyntax-only -verify -DARM %s
// RUN: %clang_cc1 -triple arm64-unknown-linux -fsyntax-only -verify -DARM %s

#define AS1 __attribute__((address_space(1)))

void default_as(char *a, char *b) {
#ifdef ARM
  __clear_cache(a, b);
#else
  __builtin___clear_cache(a, b);
#endif
}

void non_default_as(AS1 char *a, AS1 char *b) {
#ifdef ARM
  __clear_cache(a, b); // expected-error {{changes address space of pointer}}
#else
  __builtin___clear_cache(a, b); // expected-error {{changes address space of pointer}}
#endif
}
