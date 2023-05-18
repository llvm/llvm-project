// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// PR4381
template<class T> struct X {};
template<typename T> struct Y : public X<T>::X { };

// PR4621
class A1 {
  A1(int x) {}
};
template<class C> class B1 : public A1 {
  B1(C x) : A1(x.x) {}
};
class A2 { A2(int x, int y); };
template <class C> class B2 {
  A2 x;
  B2(C x) : x(x.x, x.y) {}
};
template <class C> class B3 {
  C x;
  B3() : x(1,2) {}
};

// PR4627
template<typename _Container> class insert_iterator {
    _Container* container;
    insert_iterator(_Container& __x) : container(&__x) {}
};

// PR4763
template<typename T> struct s0 {};
template<typename T> struct s0_traits {};
template<typename T> struct s1 : s0<typename s0_traits<T>::t0> {
  s1() {}
};

// PR6062
namespace PR6062 {
  template <typename T>
  class A : public T::type
  {
    A() : T::type()
    {  
    }
    
    template <typename U>
    A(U const& init)
      : T::type(init)
    { }

    template<typename U>
    A(U& init) : U::other_type(init) { }
  };
}

template<typename T, typename U>
struct X0 : T::template apply<U> {
  X0(int i) : T::template apply<U>(i) { }
};

// PR7698
namespace PR7698 {
  template<typename Type>
  class A {
    char mA[sizeof(Type *)];
    A(): mA() {}
  };
}
