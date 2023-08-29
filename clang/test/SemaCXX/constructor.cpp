// RUN: %clang_cc1 -fsyntax-only -verify %s 
typedef int INT;

class Foo {
  Foo();
  (Foo)(float) { }
  explicit Foo(int); // expected-note {{previous declaration is here}}
  Foo(const Foo&);

  ((Foo))(INT); // expected-error{{cannot be redeclared}}

  Foo(Foo foo, int i = 17, int j = 42); // expected-error{{copy constructor must pass its first argument by reference}}

  static Foo(short, short); // expected-error{{constructor cannot be declared 'static'}}
  virtual Foo(double); // expected-error{{constructor cannot be declared 'virtual'}}
  Foo(long) const; // expected-error{{'const' qualifier is not allowed on a constructor}}
  
  int Foo(int, int); // expected-error{{constructor cannot have a return type}}

  volatile Foo(float); // expected-error{{constructor cannot have a return type}}
};

Foo::Foo(const Foo&) { }

typedef struct {
  int version;
} Anon;
extern const Anon anon;
extern "C" const Anon anon2;

// PR3188: The extern declaration complained about not having an appropriate
// constructor.
struct x;
extern x a;

// A similar case.
struct y {
  y(int);
};
extern y b;

struct Length {
  Length l() const { return *this; }
};

struct mmst_reg{
 char mmst_reg[10];
};

// PR3948
namespace PR3948 {
// PR3948
class a {
  public:
  int b(int a());
};
int x();
void y() {
  a z; z.b(x);
}
}

namespace A {
  struct S {
    S();
    S(int);
    void f1();
    void f2();
    operator int ();
    ~S();
  };
}

A::S::S() {}

void A::S::f1() {}

struct S {};

A::S::S(int) {}

void A::S::f2() {}

A::S::operator int() { return 1; }

A::S::~S() {}

namespace PR38286 {
  // FIXME: It'd be nice to give more consistent diagnostics for these cases
  // (but they're all failing for somewhat different reasons...).
  template<typename> struct A;
  template<typename T> A<T>::A() {} // expected-error {{incomplete type 'A' named in nested name specifier}}
  /*FIXME: needed to recover properly from previous error*/;
  template<typename> struct B;
  template<typename T> void B<T>::f() {} // expected-error {{out-of-line definition of 'f' from class 'B<type-parameter-0-0>'}}
  template<typename> struct C; // expected-note {{non-type declaration found}}
  template<typename T> C<T>::~C() {} // expected-error {{identifier 'C' after '~' in destructor name does not name a type}}
}
