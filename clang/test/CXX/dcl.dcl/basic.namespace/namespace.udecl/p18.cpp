// RUN: %clang_cc1 -std=c++11 -verify %s

struct Public {} public_;
struct Protected {} protected_;
struct Private {} private_;

class A {
public:
  A(Public);
  void f(Public);

protected:
  A(Protected); // expected-note {{protected here}}
  void f(Protected);

private:
  A(Private); // expected-note 4{{private here}}
  void f(Private); // expected-note {{private here}}

  friend void Friend();
};

class B : private A {
  using A::A; // ok
  using A::f; // expected-error {{private member}}

  void f() {
    B a(public_);
    B b(protected_);
    B c(private_); // expected-error {{private}}
  }

  B(Public p, int) : B(p) {}
  B(Protected p, int) : B(p) {}
  B(Private p, int) : B(p) {} // expected-error {{private}}
};

class C : public B {
  C(Public p) : B(p) {}
  // There is no access check on the conversion from derived to base here;
  // protected constructors of A act like protected constructors of B.
  C(Protected p) : B(p) {}
  C(Private p) : B(p) {} // expected-error {{private}}
};

void Friend() {
  // There is no access check on the conversion from derived to base here.
  B a(public_);
  B b(protected_);
  B c(private_);
}

void NonFriend() {
  B a(public_);
  B b(protected_); // expected-error {{protected}}
  B c(private_); // expected-error {{private}}
}

namespace ProtectedAccessFromMember {
namespace a {
  struct ES {
  private:
    ES(const ES &) = delete;
  protected:
    ES(const char *);
  };
}
namespace b {
  struct DES : a::ES {
    DES *f();
  private:
    using a::ES::ES;
  };
}
b::DES *b::DES::f() { return new b::DES("foo"); }

}
