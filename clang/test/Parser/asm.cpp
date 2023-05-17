// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

int foo1 asm ("bar1");
int foo2 asm (L"bar2"); // expected-error {{cannot use wide string literal in 'asm'}}
int foo3 asm (u8"bar3"); // expected-error {{cannot use unicode string literal in 'asm'}}
int foo4 asm (u"bar4"); // expected-error {{cannot use unicode string literal in 'asm'}}
int foo5 asm (U"bar5"); // expected-error {{cannot use unicode string literal in 'asm'}}
int foo6 asm ("bar6"_x); // expected-error {{string literal with user-defined suffix cannot be used here}}
int foo6 asm ("" L"bar7"); // expected-error {{cannot use wide string literal in 'asm'}}

void f() {
  [[]] asm("");
  [[gnu::deprecated]] asm(""); // expected-warning {{'deprecated' attribute ignored}}
}
