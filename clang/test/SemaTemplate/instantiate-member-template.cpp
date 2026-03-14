// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T>
struct X0 {
  template<typename U> T f0(U);
  template<typename U> U& f1(T*, U); // expected-error{{pointer to a reference}} \
                                     // expected-note{{candidate}}
};

X0<int> x0i;
X0<void> x0v;
X0<int&> x0ir; // expected-note{{instantiation}}

void test_X0(int *ip, double *dp) {
  X0<int> xi;
  int i1 = xi.f0(ip);
  double *&dpr = xi.f1(ip, dp);
  xi.f1(dp, dp); // expected-error{{no matching}}

  X0<void> xv;
  double *&dpr2 = xv.f1(ip, dp);
}

template<typename T>
struct X1 {
  template<typename U>
  struct Inner0 {
    U x; 
    T y; // expected-error{{void}}
  };

  template<typename U>
  struct Inner1 {
    U x; // expected-error{{void}}
    T y;
  };
  
  template<typename U>
  struct Inner2 {
    struct SuperInner {
      U z; // expected-error{{void}}
    };
  };
  
  template<typename U>
  struct Inner3 {
    void f0(T t, U u) { // expected-note{{passing argument to parameter 't' here}}
      (void)(t + u); // expected-error{{invalid operands}}
    }
    
    template<typename V>
    V f1(T t, U u, V) {
      return t + u; // expected-error{{cannot initialize return object}}
    }
  };
  
  template<typename U>
  struct Inner4;
};

template<typename T>
template<typename U>
struct X1<T>::Inner4 {
  template<typename V>
  V f2(T t, U u, V);
  
  static U value;
};

template<typename T>
template<typename U>
U X1<T>::Inner4<U>::value; // expected-error{{reference variable}}

template<typename T>
template<typename U>
template<typename V>
V X1<T>::Inner4<U>::f2(T t, U u, V) {
  return t + u; // expected-error{{cannot initialize return object}}
}

void test_X1(int *ip, int i, double *dp) {
  X1<void>::Inner0<int> *xvip; // okay
  X1<void>::Inner0<int> xvi; // expected-note{{instantiation}}
  
  X1<int>::Inner1<void> *xivp; // okay
  X1<int>::Inner1<void> xiv; // expected-note{{instantiation}}
  
  X1<int>::Inner2<void>::SuperInner *xisivp; // okay
  X1<int>::Inner2<void>::SuperInner xisiv; // expected-note{{instantiation}}
  
  X1<int*>::Inner3<int> id3;
  id3.f0(ip, i);
  id3.f0(dp, i); // expected-error{{cannot initialize a parameter of type 'int *' with an lvalue of type 'double *'}}
  id3.f1(ip, i, ip);
  id3.f1(ip, i, dp); // expected-note{{instantiation}}
  
  X1<int*>::Inner3<double*> id3b;
  id3b.f0(ip, dp); // expected-note{{instantiation}}
  
  X1<int*>::Inner4<int> id4;
  id4.f2(ip, i, dp); // expected-note{{instantiation}}
  
  X1<int*>::Inner4<int>::value = 17;
  i = X1<int*>::Inner4<int&>::value; // expected-note{{instantiation}}
}


template<typename T>
struct X2 {
  template<T *Ptr> // expected-error{{pointer to a reference}}
  struct Inner;
  
  template<T Value> // expected-error{{cannot have type 'float'}}
  struct Inner2;
};

X2<int&> x2a; // expected-note{{instantiation}}
X2<float> x2b; // expected-note{{instantiation}}

namespace N0 {
  template<typename T>
  struct X0 { };
  
  struct X1 {
    template<typename T> void f(X0<T>& vals) { g(vals); }
    template<typename T> void g(X0<T>& vals) { }
  };
  
  void test(X1 x1, X0<int> x0i, X0<long> x0l) {
    x1.f(x0i);
    x1.f(x0l);
  }  
}

namespace PR6239 {
  template <typename T>  
  struct X0 {  
    class type {
      typedef T E;    
      template <E e>  // subsitute T for E and bug goes away
      struct sfinae {  };  
      
      template <class U>  
      typename sfinae<&U::operator=>::type test(int);  
    };
  };

  template <typename T>  
  struct X1 {  
    typedef T E;    
    template <E e>  // subsitute T for E and bug goes away
    struct sfinae {  };  
    
    template <class U>  
    typename sfinae<&U::operator=>::type test(int);  
  };

}

namespace PR7587 {
  template<typename> class X0;
  template<typename> struct X1;
  template<typename> class X2;

  template<typename T> class X3
  {
    template<
      template<typename> class TT,
      typename U = typename X1<T>::type
    > 
    struct Inner {
      typedef X2<TT<typename X1<T>::type> > Type;
    };

    const typename Inner<X0>::Type minCoeff() const;
  };

  template<typename T> class X3<T*>
  {
    template<
      template<typename> class TT,
      typename U = typename X1<T>::type
    > 
    struct Inner {
      typedef X2<TT<typename X1<T>::type> > Type;
    };

    const typename Inner<X0>::Type minCoeff() const;
  };

}

namespace PR7669 {
  template<class> struct X {
    template<class> struct Y {
      template<int,class> struct Z;
      template<int Dummy> struct Z<Dummy,int> {};
    };
  };

  void a()
  {
    X<int>::Y<int>::Z<0,int>();
  }
}

namespace PR8489 {
  template <typename CT>
  class C {
    template<typename FT>
    void F() {} // expected-note{{FT}}
  };
  void f() {
    C<int> c;
    c.F(); // expected-error{{no matching member function}}
  }
}

namespace rdar8986308 {
  template <bool> struct __static_assert_test;
  template <> struct __static_assert_test<true> {};
  template <unsigned> struct __static_assert_check {};

  namespace std {

    template <class _Tp, class _Up>
    struct __has_rebind
    {
    private:
      struct __two {char _; char __;};
      template <class _Xp> static __two __test(...);
      template <class _Xp> static char __test(typename _Xp::template rebind<_Up>* = 0);
    public:
      static const bool value = sizeof(__test<_Tp>(0)) == 1;
    };

  }

  template <class T> struct B1 {};

  template <class T>
  struct B
  {
    template <class U> struct rebind {typedef B1<U> other;};
  };

  template <class T, class U> struct D1 {};

  template <class T, class U>
  struct D
  {
    template <class V> struct rebind {typedef D1<V, U> other;};
  };

  int main()
  {
    typedef __static_assert_check<sizeof(__static_assert_test<((std::__has_rebind<B<int>, double>::value))>)> __t64;
    typedef __static_assert_check<sizeof(__static_assert_test<((std::__has_rebind<D<char, int>, double>::value))>)> __t64;
  }

}
