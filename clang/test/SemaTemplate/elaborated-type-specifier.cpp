// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR6915 {
  template <typename T>
  class D {
    enum T::X v; // expected-error{{use of 'X' with tag type that does not match previous declaration}} \
    // expected-error{{no enum named 'X' in 'PR6915::D3'}}
  };

  struct D1 {
    enum X { value };
  };
  struct D2 { 
    class X { }; // expected-note{{previous use is here}}
  };
  struct D3 { };

  template class D<D1>;
  template class D<D2>; // expected-note{{in instantiation of}}
  template class D<D3>; // expected-note{{in instantiation of}}
}

template<typename T>
struct DeclOrDef {
  enum T::foo; // expected-error{{nested name specifier for a declaration cannot depend on a template parameter}}
  enum T::bar { // expected-error{{nested name specifier for a declaration cannot depend on a template parameter}}
    value 
  };
};

namespace PR6649 {
  template <typename T> struct foo { 
    class T::bar;  // expected-error{{nested name specifier for a declaration cannot depend on a template parameter}}
    class T::bar { int x; }; // expected-error{{nested name specifier for a declaration cannot depend on a template parameter}}
  };
}

namespace rdar8568507 {
  template <class T> struct A *makeA(T t);
}
