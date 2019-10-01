// RUN: %clang_cc1 -fsyntax-only -verify %s

#if !__has_extension(gnu_asm)
#error Extension 'gnu_asm' should be available by default
#endif

void f1() {
  // PR7673: Some versions of GCC support an empty clobbers section.
  asm ("ret" : : :);
}

void f2() {
  asm("foo" : "=r" (a)); // expected-error {{use of undeclared identifier 'a'}}
  asm("foo" : : "r" (b)); // expected-error {{use of undeclared identifier 'b'}} 

  asm const (""); // expected-warning {{ignored const qualifier on asm}}
  asm volatile ("");
  asm restrict (""); // expected-warning {{ignored restrict qualifier on asm}}
  // FIXME: Once GCC supports _Atomic, check whether it allows this.
  asm _Atomic (""); // expected-warning {{ignored _Atomic qualifier on asm}}
}


// rdar://5952468
__asm ; // expected-error {{expected '(' after 'asm'}}

// <rdar://problem/10465079> - Don't crash on wide string literals in 'asm'.
int foo asm (L"bar"); // expected-error {{cannot use wide string literal in 'asm'}}

asm() // expected-error {{expected string literal in 'asm'}}
// expected-error@-1 {{expected ';' after top-level asm block}}

asm(; // expected-error {{expected string literal in 'asm'}}

asm("") // expected-error {{expected ';' after top-level asm block}}

// Unterminated asm strings at the end of the file were causing us to crash, so
// this needs to be last. rdar://15624081
// expected-warning@+3 {{missing terminating '"' character}}
// expected-error@+2 {{expected string literal in 'asm'}}
// expected-error@+1 {{expected ';' after top-level asm block}}
asm("
