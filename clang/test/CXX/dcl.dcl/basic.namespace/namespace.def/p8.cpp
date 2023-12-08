// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// Fun things you can do with inline namespaces:

inline namespace X {
  void f1(); // expected-note {{'f1' declared here}}

  inline namespace Y {
    void f2();

    template <typename T> class C {};
  }

  // Specialize and partially specialize somewhere else.
  template <> class C<int> {};
  template <typename T> class C<T*> {};
}

// Qualified and unqualified lookup as if member of enclosing NS.
void foo1() {
  f1();
  ::f1();
  X::f1();
  Y::f1(); // expected-error {{no member named 'f1' in namespace 'Y'; did you mean simply 'f1'?}}

  f2();
  ::f2();
  X::f2();
  Y::f2();
}

template <> class C<float> {};
template <typename T> class C<T&> {};

template class C<double>;


// As well as all the fun with ADL.

namespace ADL {
  struct Outer {};

  inline namespace IL {
    struct Inner {};

    void fo(Outer);
  }

  void fi(Inner);

  inline namespace IL2 {
    void fi2(Inner);
  }
}

void foo2() {
  ADL::Outer o;
  ADL::Inner i;
  fo(o);
  fi(i);
  fi2(i);
}

// Let's not forget overload sets.
struct Distinct {};
inline namespace Over {
  void over(Distinct);
}
void over(int);

void foo3() {
  Distinct d;
  ::over(d);
}

// Don't forget to do correct lookup for redeclarations.
namespace redecl { inline namespace n1 {

  template <class Tp> class allocator;

  template <>
  class allocator<void>
  {
  public:
      typedef const void* const_pointer;
  };

  template <class Tp>
  class allocator
  {
  public:
      typedef Tp& reference;
  
      void allocate(allocator<void>::const_pointer = 0);
  };

} }

// Normal redeclarations (not for explicit instantiations or
// specializations) are distinct in an inline namespace vs. not in an
// inline namespace.
namespace redecl2 { 
  inline namespace n1 {
    void f(int) { }
    struct X1 { };
    template<typename T> void f(T) { }
    template<typename T> struct X2 { };
    int i = 71;
    enum E { e };
  }

  void f(int) { }
  struct X1 { };
  template<typename T> void f(T) { }
  template<typename T> struct X2 { };
  int i = 71;
  enum E { e };
}
