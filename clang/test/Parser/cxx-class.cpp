// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
class C;
class C {
public:
protected:
  typedef int A,B;
  static int sf(), u;

  struct S {};
  enum {}; // expected-warning{{declaration does not declare anything}}
  int; // expected-warning {{declaration does not declare anything}}
  int : 1, : 2;

public:
  void m0() {}; // ok, one extra ';' is permitted
  void m1() {}
  ; // ok, one extra ';' is permitted
  void m() {
    int l = 2;
  };; // expected-warning{{extra ';' after member function definition}}

  template<typename T> void mt(T) { }
  ;
  ; // expected-warning{{extra ';' inside a class}}

  virtual int vf() const volatile = 0;
  
private:
  int x,f(),y,g();
  inline int h();
  static const int sci = 10;
  mutable int mi;
};
void glo()
{
  struct local {};
}

// PR3177
typedef union {
  __extension__ union {
    int a;
    float b;
  } y;
} bug3177;

// check that we don't consume the token after the access specifier 
// when it's not a colon
class D {
public // expected-error{{expected ':'}}
  int i;
};

// consume the token after the access specifier if it's a semicolon 
// that was meant to be a colon
class E {
public; // expected-error{{expected ':'}}
  int i;
};

class F {
    int F1 { return 1; } // expected-error{{function definition does not declare parameters}}
    void F2 {} // expected-error{{function definition does not declare parameters}}
    typedef int F3() { return 0; } // expected-error{{function definition declared 'typedef'}}
    typedef void F4() {} // expected-error{{function definition declared 'typedef'}}
};

namespace ctor_error {
  class Foo {};
  // By [class.qual]p2, this is a constructor declaration.
  Foo::Foo (F) = F(); // expected-error{{does not match any declaration in 'ctor_error::Foo'}}

  class Ctor { // expected-note{{not complete until the closing '}'}}
    Ctor(f)(int); // ok
    Ctor(g(int)); // ok
    Ctor(x[5]); // expected-error{{incomplete type}}

    Ctor(UnknownType *); // expected-error{{unknown type name 'UnknownType'}}
    void operator+(UnknownType*); // expected-error{{unknown type name 'UnknownType'}}
  };

  Ctor::Ctor (x) = { 0 }; // \
    // expected-error{{qualified reference to 'Ctor' is a constructor name}}

  Ctor::Ctor(UnknownType *) {} // \
    // expected-error{{unknown type name 'UnknownType'}}
  void Ctor::operator+(UnknownType*) {} // \
    // expected-error{{unknown type name 'UnknownType'}}
}

// PR13775: Don't assert here.
namespace PR13775 {
  class bar
  {
   public:
    void foo ();
    void baz ();
  };
  void bar::foo ()
  {
    baz x(); // expected-error 3{{}}
  }
}

// PR11109 must appear at the end of the source file
class pr11109r3 { // expected-note{{to match this '{'}}
  public // expected-error{{expected ':'}} expected-error{{expected '}'}} expected-error{{expected ';' after class}}
