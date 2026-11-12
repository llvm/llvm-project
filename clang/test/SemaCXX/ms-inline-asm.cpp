// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple i386-apple-darwin10 -fasm-blocks -verify

struct A {
  int a1;
  int a2;
  struct B {
    int b1;
    int b2;
    enum { kValue = 42 };
  } a3;
  struct {
    int indirect_field;
  };
};

namespace asdf {
A a_global;
}

// The parser combines adjacent __asm blocks into one. Avoid that by calling
// this.
void split_inline_asm_call();

void test_field_lookup() {
  __asm mov eax, asdf::a_global.a3.b2
  split_inline_asm_call();

  // FIXME: These diagnostics are crap.

  // expected-error@+1 {{undeclared label}}
  __asm mov eax, asdf::a_global.not_a_field.b2
  split_inline_asm_call();

  // expected-error@+1 {{undeclared label}}
  __asm mov eax, asdf::a_global.a3.not_a_field
  split_inline_asm_call();

  __asm mov eax, A::B::kValue
  split_inline_asm_call();

  // expected-error@+1 {{undeclared label}}
  __asm mov eax, asdf::a_global.a3.kValue
  split_inline_asm_call();

  __asm mov eax, asdf :: a_global.a3.b2
  split_inline_asm_call();

  __asm mov eax, asdf::a_global . a3 . b2
  split_inline_asm_call();

  __asm mov eax, asdf::a_global.indirect_field
  split_inline_asm_call();
}
