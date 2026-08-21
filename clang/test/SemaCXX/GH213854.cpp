// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace reduced {
union Union {
  class A {
    virtual void foo();
  };
  class B : public A {
  };
  void B::foo() {} // expected-error {{non-friend class member 'foo' cannot have a qualified name}}
};

static_assert(!__is_polymorphic(Union), "");

void uni(void (*fn)(Union), Union arg1) {
  fn(arg1);
}

struct Struct {
  class A {
    virtual void foo();
  };
  class B : public A {
  };
  void B::foo() {} // expected-error {{non-friend class member 'foo' cannot have a qualified name}}
};

static_assert(!__is_polymorphic(Struct), "");
} // namespace reduced

// Verbatim reproducer from GH213854; the missing closing brace is intentional.
union Union { // expected-note {{to match this '{'}}
  class A {
  virtual void foo();
  };
  class B : public A {
  };
  void B::foo() {} // expected-error {{non-friend class member 'foo' cannot have a qualified name}}
void uni(void (*fn)(union Union), union Union arg1) {
    fn(arg1);
}
// expected-error {{expected '}'}} expected-error@-1 {{expected ';' after union}}