// RUN: %clang_cc1 -fsyntax-only -verify %s
int foo(int);

namespace N {
  void f1() {
    void foo(int); // okay
    void bar(int); // expected-note 2{{previous declaration is here}}
  }

  void foo(int); // expected-note 3{{previous declaration is here}}

  void f2() {
    int foo(int); // expected-error {{functions that differ only in their return type cannot be overloaded}}
    int bar(int); // expected-error {{functions that differ only in their return type cannot be overloaded}}
    int baz(int); // expected-note {{previous declaration is here}}

    {
      int foo;
      int bar;
      int baz;
      {
        float foo(int); // expected-error {{functions that differ only in their return type cannot be overloaded}}
        float bar(int); // expected-error {{functions that differ only in their return type cannot be overloaded}}
        float baz(int); // expected-error {{functions that differ only in their return type cannot be overloaded}}
      }
    }
  }

  void f3() {
    int foo(float);
    {
      float foo(int); // expected-error {{functions that differ only in their return type cannot be overloaded}}
    }
  }
}

class A {
 void typocorrection(); // expected-note {{'typocorrection' declared here}}
};

void A::Notypocorrection() { // expected-error {{out-of-line definition of 'Notypocorrection' does not match any declaration in 'A'; did you mean 'typocorrection'}}
}


namespace test0 {
  void dummy() {
    void Bar(); // expected-note {{'Bar' declared here}}
    class A {
      friend void bar(); // expected-error {{cannot define friend function 'bar' in a local class definition; did you mean 'Bar'}}
    };
  }
}


class B {
 void typocorrection(const int); // expected-note {{'typocorrection' declared here}}
 void typocorrection(double);
};

void B::Notypocorrection(int) { // expected-error {{out-of-line definition of 'Notypocorrection' does not match any declaration in 'B'; did you mean 'typocorrection'}}
}

struct X { int f(); };
struct Y : public X {}; // expected-note {{defined here}}
int Y::f() { return 3; } // expected-error {{out-of-line definition of 'f' does not match any declaration in 'Y'}}

namespace test1 {
struct Foo {
  class Inner { };
};
}

class Bar { // expected-note {{defined here}}
  void f(test1::Foo::Inner foo) const; // expected-note {{member declaration does not match because it is const qualified}}
};

using test1::Foo;

void Bar::f(Foo::Inner foo) { // expected-error {{out-of-line definition of 'f' does not match any declaration in 'Bar'}}
  (void)foo;
}

class Crash { // expected-note {{defined here}}
              // expected-note@-1 {{defined here}}
 public:
  void GetCart(int count) const;
};
// This out-of-line definition was fine...
void Crash::cart(int count) const {} // expected-error {{out-of-line definition of 'cart' does not match any declaration in 'Crash'}}
// ...while this one crashed clang
void Crash::chart(int count) const {} // expected-error {{out-of-line definition of 'chart' does not match any declaration in 'Crash'}}

class TestConst { // expected-note {{defined here}}
                  // expected-note@-1 {{defined here}}
 public:
  int getit() const; // expected-note {{member declaration does not match because it is const qualified}}
  void setit(int); // expected-note {{member declaration does not match because it is not const qualified}}
};

int TestConst::getit() { // expected-error {{out-of-line definition of 'getit' does not match any declaration in 'TestConst'}}
  return 1;
}

void TestConst::setit(int) const { // expected-error {{out-of-line definition of 'setit' does not match any declaration in 'TestConst'}}
}

struct J { int typo() const; }; // expected-note {{defined here}}
int J::typo_() { return 3; } // expected-error {{out-of-line definition of 'typo_' does not match any declaration in 'J'}}

// Ensure we correct the redecl of Foo::isGood to Bar::Foo::isGood and not
// Foo::IsGood even though Foo::IsGood is technically a closer match since it
// already has a body. Also make sure Foo::beEvil is corrected to Foo::BeEvil
// since it is a closer match than Bar::Foo::beEvil and neither have a body.
namespace redecl_typo {
namespace Foo {
  bool IsGood() { return false; }
  void BeEvil(); // expected-note {{'BeEvil' declared here}}
}
namespace Bar {
  namespace Foo {
    bool isGood(); // expected-note {{'Bar::Foo::isGood' declared here}}
    void beEvil();
  }
}
bool Foo::isGood() { // expected-error {{out-of-line definition of 'isGood' does not match any declaration in namespace 'redecl_typo::Foo'; did you mean 'Bar::Foo::isGood'?}}
  return true;
}
void Foo::beEvil() {} // expected-error {{out-of-line definition of 'beEvil' does not match any declaration in namespace 'redecl_typo::Foo'; did you mean 'BeEvil'?}}
}

struct CVQualFun { // expected-note {{defined here}}
  void func(int a, int &b); // expected-note {{type of 2nd parameter of member declaration does not match definition ('int &' vs 'int')}}
};

void CVQualFun::func(const int a, int b) {} // expected-error {{out-of-line definition of 'func' does not match any declaration in 'CVQualFun'}}
