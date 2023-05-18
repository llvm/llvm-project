// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

extern "C" { void f(bool); }

namespace std {
  using ::f;
  inline void f() { return f(true); }
}

namespace M {
  void f(float);
}

namespace N {
  using M::f;
  void f(int) { } // expected-note{{previous}}
  
  void f(int) { } // expected-error{{redefinition}}
}

namespace N {
  void f(double);
  void f(long);
}

struct X0 {
  void operator()(int);
  void operator()(long);
};

struct X1 : X0 {
  void operator()(float&);
  using X0::operator();
  
  void test() {
    (*this)(1);
  }
};

struct A { void f(); };
struct B : A { };
class C : B { using B::f; };

// PR5751: Resolve overloaded functions through using decls.
namespace O {
  void f(int i);
  void f(double d);
}
namespace P {
  void f();
  void g(void (*ptr)(int));
  using O::f;
  void test() {
    f();
    f(1);
    void (*f_ptr1)(double) = f;
    void (*f_ptr2)() = f;
    g(f);
  }
}

// Make sure that ADL can find names brought in by using decls.
namespace test0 {
  namespace ns {
    class Foo {};
    
    namespace inner {
      void foo(char *); // expected-note {{no known conversion}} 
    }

    using inner::foo;
  }

  void test(ns::Foo *p) {
    foo(*p); // expected-error {{no matching function for call to 'foo'}}
  }
}

// Redeclarations!
namespace test1 {
  namespace ns0 { struct Foo {}; }
  namespace A { void foo(ns0::Foo *p, int y, int z); }
  namespace ns2 { using A::foo; }
  namespace ns1 { struct Bar : ns0::Foo {}; }
  namespace A { void foo(ns0::Foo *p, int y, int z = 0); } // expected-note {{candidate}}
  namespace ns1 { using A::foo; }
  namespace ns2 { struct Baz : ns1::Bar {}; }
  namespace A { void foo(ns0::Foo *p, int y = 0, int z); }

  void test(ns2::Baz *p) {
    foo(p, 0, 0); // okay!
    foo(p, 0); // should be fine!
    foo(p); // expected-error {{no matching function}}
  }
}

namespace test2 {
  namespace ns { int foo; }
  template <class T> using ns::foo; // expected-error {{cannot template a using declaration}}

  // PR8022
  struct A {
    template <typename T> void f(T);
  };
  class B : A {
    template <typename T> using A::f<T>; // expected-error {{cannot template a using declaration}}
  };
}

// PR8756
namespace foo
{
  class Class1; // expected-note{{forward declaration}}
  class Class2
  {
    using ::foo::Class1::Function; // expected-error{{incomplete type 'foo::Class1' named in nested name specifier}}
  };
}

// Don't suggest non-typenames for positions requiring typenames.
namespace using_suggestion_tyname_val {
namespace N { void FFF() {} }
using typename N::FFG; // expected-error {{no member named 'FFG' in namespace 'using_suggestion_tyname_val::N'}}
}

namespace using_suggestion_member_tyname_val {
class CCC { public: void AAA() { } };
class DDD : public CCC { public: using typename CCC::AAB; }; // expected-error {{no member named 'AAB' in 'using_suggestion_member_tyname_val::CCC'}}
}

namespace using_suggestion_tyname_val_dropped_specifier {
void FFF() {}
namespace N { }
using typename N::FFG; // expected-error {{no member named 'FFG' in namespace 'using_suggestion_tyname_val_dropped_specifier::N'}}
}

// Currently hints aren't provided to drop out the incorrect M::.
namespace using_suggestion_ty_dropped_nested_specifier {
namespace N {
class AAA {}; // expected-note {{'N::AAA' declared here}}
namespace M { }
}
using N::M::AAA; // expected-error {{no member named 'AAA' in namespace 'using_suggestion_ty_dropped_nested_specifier::N::M'; did you mean 'N::AAA'?}}
}

namespace using_suggestion_tyname_ty_dropped_nested_specifier {
namespace N {
class AAA {}; // expected-note {{'N::AAA' declared here}}
namespace M { }
}
using typename N::M::AAA; // expected-error {{no member named 'AAA' in namespace 'using_suggestion_tyname_ty_dropped_nested_specifier::N::M'; did you mean 'N::AAA'?}}
}

namespace using_suggestion_val_dropped_nested_specifier {
namespace N {
void FFF() {} // expected-note {{'N::FFF' declared here}}
namespace M { }
}
using N::M::FFF; // expected-error {{no member named 'FFF' in namespace 'using_suggestion_val_dropped_nested_specifier::N::M'; did you mean 'N::FFF'?}}
}

namespace UsingDeclVsHiddenName {
  namespace A {
    enum HiddenTag1 {}; // expected-note {{previous use is here}}
    enum HiddenTag2 {}; // expected-note {{target}}
    int HiddenFn1; // expected-note {{target}}
    int HiddenFn2; // expected-note {{target}}
    int HiddenLocalExtern1;
    int HiddenLocalExtern2;
  }

  namespace B {
    using A::HiddenTag1;
    using A::HiddenFn1; // expected-note {{using declaration}}
    using A::HiddenLocalExtern1;

    struct S {
      friend struct HiddenTag1; // expected-error {{tag type that does not match previous}}
      friend struct HiddenTag2; // expected-note {{conflicting declaration}}
      friend void HiddenFn1(); // expected-error {{cannot befriend target of using declaration}}
      friend void HiddenFn2(); // expected-note {{conflicting declaration}}
      void f() {
        // OK, these are not in the scope of namespace B, even though they're
        // members of the namespace.
        void HiddenLocalExtern1();
        void HiddenLocalExtern2();
      }
    };

    using A::HiddenTag2; // expected-error {{conflicts with declaration already in scope}}
    using A::HiddenFn2; // expected-error {{conflicts with declaration already in scope}}
    using A::HiddenLocalExtern2;
  }
}

namespace PR19171 {
  struct Z {
    Z();
  };

  typedef struct {
    Z i;
  } S;

  struct Y : S {
    using S::S;
#if __cplusplus < 201103L
    // expected-error@-2 {{no member named 'S' in 'PR19171::S'}}
#endif
  };

  // [namespace.udecl]p3: In a using-declaration used as a member-declaration,
  // the nested-name-specifier shall name a base class of the class being defined.
  // If such a using-declaration names a constructor, the nested-name-specifier
  // shall name a direct base class of the class being defined;

  struct B_blah { };
  struct C_blah : B_blah { C_blah(int); }; // expected-note 0-1{{declared here}}
  struct D1 : C_blah {
    // FIXME: We should be able to correct this in C++11 mode.
    using B_blah::C_blah; // expected-error-re {{no member named 'C_blah' in 'PR19171::B_blah'{{$}}}}
  };
  struct D2 : C_blah {
    // Somewhat bizarrely, this names the injected-class-name of B_blah within
    // C_blah, and is valid.
    using C_blah::B_blah;
  };
  struct D3 : C_blah {
    using C_blah::D_blah;
#if __cplusplus < 201103L
    // expected-error-re@-2 {{no member named 'D_blah' in 'PR19171::C_blah'{{$}}}}
#else
    // expected-error@-4 {{no member named 'D_blah' in 'PR19171::C_blah'; did you mean 'C_blah'?}}
#endif
  };
#if __cplusplus >= 201103L
  D3 d3(0); // ok
#endif

  struct E { };
  struct EE { int EE; };
  struct F : E {
    using E::EE; // expected-error-re {{no member named 'EE' in 'PR19171::E'{{$}}}}
  };

  struct TypoDuplicate { // expected-note 0-4{{here}}
    TypoDuplicate(int);
    void foobar(); // expected-note 2{{here}}
  };
  struct TypoDuplicateDerived1 : TypoDuplicate {
#if __cplusplus >= 201103L
    using TypoDuplicate::TypoFuplicate; // expected-error {{did you mean 'TypoDuplicate'}} expected-note {{previous}}
    using TypoDuplicate::TypoDuplicate; // expected-error {{redeclaration}}
#endif
    using TypoDuplicate::goobar; // expected-error {{did you mean 'foobar'}} expected-note {{previous}}
    using TypoDuplicate::foobar; // expected-error {{redeclaration}}
  };
  struct TypoDuplicateDerived2 : TypoDuplicate {
#if __cplusplus >= 201103L
    using TypoFuplicate::TypoDuplicate; // expected-error {{did you mean 'TypoDuplicate'}} expected-note {{previous}}
    using TypoDuplicate::TypoDuplicate; // expected-error {{redeclaration}}
#endif
  };
  struct TypoDuplicateDerived3 : TypoDuplicate {
#if __cplusplus >= 201103L
    // FIXME: Don't suggest a correction that would lead to a redeclaration
    // error here... or at least diagnose the error.
    using TypoDuplicate::TypoDuplicate;
    using TypoDuplicate::TypoFuplicate; // expected-error {{did you mean 'TypoDuplicate'}}
#endif
    using TypoDuplicate::foobar;
    using TypoDuplicate::goobar; // expected-error {{did you mean 'foobar'}}
  };
  struct TypoDuplicateDerived4 : TypoDuplicate {
#if __cplusplus >= 201103L
    using TypoDuplicate::TypoDuplicate; // expected-note {{previous}}
    using TypoFuplicate::TypoDuplicate; // expected-error {{did you mean 'TypoDuplicate'}} expected-error {{redeclaration}}
#endif
  };
}

namespace TypoCorrectTemplateMember {
  struct A {
    template<typename T> void foobar(T); // expected-note {{'foobar' declared here}}
  };
  struct B : A {
    using A::goobar; // expected-error {{no member named 'goobar' in 'TypoCorrectTemplateMember::A'; did you mean 'foobar'?}}
  };
}

namespace use_instance_in_static {
struct A { int n; };
struct B : A {
  using A::n;
  static int f() { return n; } // expected-error {{invalid use of member 'n' in static member function}}
};
}

namespace PR24030 {
  namespace X {
    class A; // expected-note {{target}}
    int i; // expected-note {{target}}
  }
  namespace Y {
    using X::A; // expected-note {{using}}
    using X::i; // expected-note {{using}}
    class A {}; // expected-error {{conflicts}}
    int i; // expected-error {{conflicts}}
  }
}

namespace PR24033 {
  extern int a; // expected-note 2{{target of using declaration}}
  void f(); // expected-note 2{{target of using declaration}}
  struct s; // expected-note 2{{target of using declaration}}
  enum e {}; // expected-note 2{{target of using declaration}}

  template<typename> extern int vt; // expected-note 2{{target of using declaration}} expected-warning 0-1{{extension}}
  template<typename> void ft(); // expected-note 2{{target of using declaration}}
  template<typename> struct st; // expected-note 2{{target of using declaration}}

  namespace X {
    using PR24033::a; // expected-note {{using declaration}}
    using PR24033::f; // expected-note {{using declaration}}
    using PR24033::s; // expected-note {{using declaration}}
    using PR24033::e; // expected-note {{using declaration}}

    using PR24033::vt; // expected-note {{using declaration}}
    using PR24033::ft; // expected-note {{using declaration}}
    using PR24033::st; // expected-note {{using declaration}}

    extern int a; // expected-error {{declaration conflicts with target of using declaration already in scope}}
    void f(); // expected-error {{declaration conflicts with target of using declaration already in scope}}
    struct s; // expected-error {{declaration conflicts with target of using declaration already in scope}}
    enum e {}; // expected-error {{declaration conflicts with target of using declaration already in scope}}

    template<typename> extern int vt; // expected-error {{declaration conflicts with target of using declaration already in scope}} expected-warning 0-1{{extension}}
    template<typename> void ft(); // expected-error {{declaration conflicts with target of using declaration already in scope}}
    template<typename> struct st; // expected-error {{declaration conflicts with target of using declaration already in scope}}
  }

  namespace Y {
    extern int a; // expected-note {{conflicting declaration}}
    void f(); // expected-note {{conflicting declaration}}
    struct s; // expected-note {{conflicting declaration}}
    enum e {}; // expected-note {{conflicting declaration}}

    template<typename> extern int vt; // expected-note {{conflicting declaration}} expected-warning 0-1{{extension}}
    template<typename> void ft(); // expected-note {{conflicting declaration}}
    template<typename> struct st; // expected-note {{conflicting declaration}}

    using PR24033::a; // expected-error {{target of using declaration conflicts with declaration already in scope}}
    using PR24033::f; // expected-error {{target of using declaration conflicts with declaration already in scope}}
    using PR24033::s; // expected-error {{target of using declaration conflicts with declaration already in scope}}
    using PR24033::e; // expected-error {{target of using declaration conflicts with declaration already in scope}}

    using PR24033::vt; // expected-error {{target of using declaration conflicts with declaration already in scope}}
    using PR24033::ft; // expected-error {{target of using declaration conflicts with declaration already in scope}}
    using PR24033::st; // expected-error {{target of using declaration conflicts with declaration already in scope}}
  }
}

namespace field_use {
struct A { int field; };
struct B : A {
  // Previously Clang rejected this valid C++11 code because it didn't look
  // through the UsingShadowDecl.
  using A::field;
#if __cplusplus < 201103L
  // expected-error@+2 {{invalid use of non-static data member 'field'}}
#endif
  enum { X = sizeof(field) };
};
}

namespace tag_vs_var {
  namespace N {
    struct X {};

    struct Y {};
    int Y;

    int Z;
  }
  using N::X;
  using N::Y;
  using N::Z;

  namespace N {
    int X;

    struct Z {};
  }
  using N::X;
  using N::Y;
  using N::Z;
}

// expected-error@+5 {{requires a qualified name}}
// expected-error@+4 {{expected ';'}}
// expected-error@+3 {{expected '}'}}
// expected-note@+2 {{to match this '{'}}
// expected-error@+1 {{expected ';'}}
template<class> struct S { using S