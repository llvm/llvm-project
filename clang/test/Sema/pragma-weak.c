// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s

void __both3(void);
#pragma weak both3 = __both3 // expected-note {{previous definition}}
void both3(void) __attribute((alias("__both3"))); // expected-error {{redefinition of 'both3'}}
void __both3(void) {}

void __a3(void) __attribute((noinline));
#pragma weak a3 = __a3 // expected-note {{previous definition}}
void a3(void) __attribute((alias("__a3"))); // expected-error {{redefinition of 'a3'}}
void __a3(void) {}
