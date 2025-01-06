// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx11 -Wc++11-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wc++11-compat %s -std=c++98
class C {
public:
  auto int errx; // expected-error {{storage class specified for a member declaration}}
#if __cplusplus <= 199711L
  // expected-warning@-2 {{'auto' storage class specifier is redundant}}
#else
  // expected-warning@-4 {{'auto' storage class specifier is not permitted in C++11, and will not be supported in future releases}}
#endif
  register int erry; // expected-error {{storage class specified for a member declaration}}
  extern int errz; // expected-error {{storage class specified for a member declaration}}

  static void sm() {
    sx = 0;
    this->x = 0; // expected-error {{invalid use of 'this' outside of a non-static member function}}
    x = 0; // expected-error {{invalid use of member 'x' in static member function}}
  }

  class NestedC {
  public:
    NestedC(int);
    void f() {
      sx = 0;
      x = 0; // expected-error {{use of non-static data member 'x' of 'C' from nested type 'NestedC'}}
      sm();
      m(); // expected-error {{call to non-static member function 'm' of 'C' from nested type 'NestedC'}}
    }
  };

  int b : 1, w : 2;
  int : 1, : 2;
  typedef int E : 1; // expected-error {{typedef member 'E' cannot be a bit-field}}
  static int sb : 1; // expected-error {{static member 'sb' cannot be a bit-field}}
  static int vs;

  typedef int func();
  func tm;
  func *ptm;
  func btm : 1; // expected-error {{bit-field 'btm' has non-integral type}}
  NestedC bc : 1; // expected-error {{bit-field 'bc' has non-integral type}}

  enum E1 { en1, en2 };

  int i = 0;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{default member initializer for non-static data member is a C++11 extension}}
#endif
  static int si = 0; // expected-error {{non-const static data member must be initialized out of line}}
  static const NestedC ci = 0; // expected-error {{static data member of type 'const NestedC' must be initialized out of line}}
  static const int nci = vs; // expected-error {{in-class initializer for static data member is not a constant expression}}
  static const int vi = 0;
  static const volatile int cvi = 0; // ok, illegal in C++11
#if __cplusplus >= 201103L
  // expected-error@-2 {{static const volatile data member must be initialized out of line}}
#endif
  static const E evi = 0;
  static const int overflow = 1000000*1000000; // cxx11-error {{in-class initializer for static data member is not a constant expression}}
                                               // expected-warning@-1 {{overflow in expression}}
  static const int overflow_shift = 1<<32; // cxx11-error {{in-class initializer for static data member is not a constant expression}}
  static const int overflow_shift2 = 1>>32; // cxx11-error {{in-class initializer for static data member is not a constant expression}}
  static const int overflow_shift3 = 1<<-1; // cxx11-error {{in-class initializer for static data member is not a constant expression}}
  static const int overflow_shift4 = 1<<-1; // cxx11-error {{in-class initializer for static data member is not a constant expression}}
  static const int overflow_shift5 = -1<<1; // cxx11-error {{in-class initializer for static data member is not a constant expression}}

  void m() {
    sx = 0;
    this->x = 0;
    y = 0;
    this = 0; // expected-error {{expression is not assignable}}
  }

  int f1(int p) {
    A z = 6;
    return p + x + this->y + z;
  }

  typedef int A;

  virtual int viv; // expected-error {{'virtual' can only appear on non-static member functions}}
  virtual static int vsif(); // expected-error {{'virtual' can only appear on non-static member functions}}
  virtual int vif();

private:
  int x,y;
  static int sx;

  mutable int mi;
  mutable int &mir; // expected-error {{'mutable' cannot be applied to references}}
  mutable void mfn(); // expected-error {{'mutable' cannot be applied to functions}}
  mutable const int mci; // expected-error {{'mutable' and 'const' cannot be mixed}}

  static const int number = 50;
  static int arr[number];
};

class C2 {
  void f() {
    static int lx;
    class LC1 {
      int m() { return lx; }
    };
    class LC2 {
      int m() { return lx; }
    };
  }
};

struct C3 {
  int i;
  mutable int j;
};
void f()
{
  const C3 c3 = { 1, 2 };
  (void)static_cast<int*>(&c3.i); // expected-error {{static_cast from 'const int *' to 'int *' is not allowed}}
  // but no error here
  (void)static_cast<int*>(&c3.j);
}

// Play with mutable a bit more, to make sure it doesn't crash anything.
mutable int gi; // expected-error {{'mutable' can only be applied to member variables}}
mutable void gfn(); // expected-error {{illegal storage class on function}}
void ogfn()
{
  mutable int ml; // expected-error {{'mutable' can only be applied to member variables}}

  // PR3020: This used to crash due to double ownership of C4.
  struct C4;
  C4; // expected-warning {{declaration does not declare anything}}
}

struct C4 {
  void f(); // expected-note{{previous declaration is here}}
  int f; // expected-error{{duplicate member 'f'}}
};

// PR5415 - don't hang!
struct S
{
  void f(); // expected-note 1 {{previous declaration}} expected-note {{previous declaration}}
  void S::f() {} // expected-error {{extra qualification on member}} expected-error {{class member cannot be redeclared}}
  void f() {} // expected-error {{class member cannot be redeclared}}
};

// Don't crash on this bogus code.
namespace pr6629 {
  template<class T1, class T2> struct foo :
    bogus<foo<T1,T2> > // expected-error {{no template named 'bogus'}}
  { };

  template<> struct foo<unknown,unknown> { // expected-error {{undeclared identifier 'unknown'}}
    template <typename U1, typename U2> struct bar {
      typedef bar type;
      static const int value = 0;
    };
  };
}

namespace PR7153 {
  class EnclosingClass {
  public:
    struct A { } mutable *member;
  };
 
  void f(const EnclosingClass &ec) {
    ec.member = 0;
  }
}

namespace PR7196 {
  struct A {
    int a;

    void f() {
      char i[sizeof(a)];
      enum { x = sizeof(i) };
      enum { y = sizeof(a) };
    }
  };
}

namespace rdar8066414 {
  class C {
    C() {}
  } // expected-error{{expected ';' after class}}
}

namespace rdar8367341 {
  float foo();
#if __cplusplus >= 201103L
  // expected-note@-2 {{declared here}}
#endif

  struct A {
#if __cplusplus <= 199711L
    static const float x = 5.0f; // expected-warning {{in-class initializer for static data member of type 'const float' is a GNU extension}}
    static const float y = foo(); // expected-warning {{in-class initializer for static data member of type 'const float' is a GNU extension}} expected-error {{in-class initializer for static data member is not a constant expression}}
#else
    static constexpr float x = 5.0f;
    static constexpr float y = foo(); // expected-error {{constexpr variable 'y' must be initialized by a constant expression}} expected-note {{non-constexpr function 'foo' cannot be used in a constant expression}}
#endif
  };
}

namespace with_anon {
struct S {
  union {
    char c;
  };
};

void f() {
    S::c; // expected-error {{invalid use of non-static data member}}
}
}

struct PR9989 { 
  static int const PR9989_Member = sizeof PR9989_Member; 
};
