// RUN: %clang_cc1 -fsyntax-only -verify %s

void foo() {
  // Test the crash case from GH177056 - this specific syntax triggered the assert
  asm("" ::: (u8"")); // expected-error {{the expression in this asm operand must be a string literal or an object with 'data()' and 'size()' member functions}}
}

void test_other_literals() {
  asm(L""); // expected-error {{cannot use wide string literal in 'asm'}}
  asm(u""); // expected-error {{cannot use unicode string literal in 'asm'}}
  asm(U""); // expected-error {{cannot use unicode string literal in 'asm'}}
  asm(u8""); // expected-error {{cannot use unicode string literal in 'asm'}}
  
  // Raw string literals should be accepted (they're still ordinary strings)
  asm(R"(nop)");
}
