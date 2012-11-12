// RUN: %clang_cc1 -fsyntax-only -verify %s

void f1() {
  // PR7673: Some versions of GCC support an empty clobbers section.
  asm ("ret" : : :);
}

void f2() {
  asm("foo" : "=r" (a)); // expected-error {{use of undeclared identifier 'a'}}
  asm("foo" : : "r" (b)); // expected-error {{use of undeclared identifier 'b'}} 
}


// rdar://5952468
__asm ; // expected-error {{expected '(' after 'asm'}}

// <rdar://problem/10465079> - Don't crash on wide string literals in 'asm'.
int foo asm (L"bar"); // expected-error {{cannot use wide string literal in 'asm'}}

