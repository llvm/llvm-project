// RUN: %clang_cc1 -std=c++2a -verify %s

void *operator new(__SIZE_TYPE__, void*);

// Check that we give a good diagnostic for an attempt to use a postfix
// operator after a unary-expression.
namespace postfix_after_unary {
  struct A { int n; };
  int &a = new A->n; // expected-error {{expression cannot be followed by a postfix '->' operator; add parentheses}}

  struct B { B(int); int operator()(int); };
  int n = new (0) (B) (int()) (int()); // expected-error {{cannot be followed by a postfix '(}} expected-error {{not a function or function pointer}}

  char x = sizeof(int)["hello"]; // expected-error {{cannot be followed by a postfix '[}}
  char y = alignof(int)["hello"]; // expected-error {{cannot be followed by a postfix '[}}
  char z = noexcept(0)["hello"]; // expected-error {{cannot be followed by a postfix '[}}
  char w = requires { x == x; }["ny"]; // expected-error {{cannot be followed by a postfix '[}}

  int f() {
  label:
    return &&label->n; // expected-error {{cannot be followed by a postfix}} expected-error {{not a structure or union}}
  }

  char k = sizeof(int) // expected-error {{expected ';'}}
  [[noreturn]] void g();
}

// Check that we do parse postfix-expression suffixes after some more unusual
// kinds of postfix-expressions (null literals and builtin calls).
namespace unusual_primary_exprs {
  int a = nullptr["foo"]; // expected-error {{array subscript is not an integer}}
  int b = __builtin_COLUMN()["sufficiently long string constant"];
  int c = __builtin_available(*)["ny"];
  static_assert(__null["foo"] == 'f'); // FIXME: Warn about converting __null to integer in array subscripting.
  static_assert(__is_standard_layout(int)["ny"] == 'y');
  static_assert(__array_rank(int[1][2])["0123"] == '2');
  static_assert(__is_lvalue_expr(a)["ny"] == 'y');
}
