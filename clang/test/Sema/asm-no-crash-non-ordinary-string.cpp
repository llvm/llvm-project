// RUN: %clang_cc1 -fsyntax-only -verify %s

void foo() {
  asm("" ::: (u8""}));
  // expected-error@-1 {{cannot use unicode string literal in 'asm'}}
  // expected-error@-2 {{expected ')'}}
}

void test_other_literals() {
  asm(L""); // expected-error {{cannot use wide string literal in 'asm'}}
  asm(u""); // expected-error {{cannot use unicode string literal in 'asm'}}
  asm(U""); // expected-error {{cannot use unicode string literal in 'asm'}}
  
  // Raw string literals should be accepted (they're still ordinary strings)
  asm(R"(nop)");
}
