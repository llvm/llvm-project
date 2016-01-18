// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class X, class Y, class Z> X f(Y,Z); // expected-note {{candidate template ignored: couldn't infer template argument 'X'}}

void g() {
  f<int,char*,double>("aa",3.0); // expected-warning{{conversion from string literal to 'char *' is deprecated}}
  f<int,char*>("aa",3.0); // Z is deduced to be double  \
                          // expected-warning{{conversion from string literal to 'char *' is deprecated}}
  f<int>("aa",3.0);       // Y is deduced to be char*, and
                          // Z is deduced to be double 
  f("aa",3.0); // expected-error{{no matching}}
}

// PR5910
namespace PR5910 {
  template <typename T>
  void Func() {}
  
  template <typename R>
  void Foo(R (*fp)());
  
  void Test() {
    Foo(Func<int>);
  }
}

// PR5949
namespace PR5949 {
  struct Bar;

  template <class Container>
  void quuz(const Container &cont) {
  }

  template<typename T>
  int Foo(Bar *b, void (*Baz)(const T &t), T * = 0) {
    return 0;
  }

  template<typename T>
  int Quux(Bar *b, T * = 0)
  {
    return Foo<T>(b, quuz);
  }
}

// PR7641
namespace PR7641 {
  namespace N2
  {
    template<class>
    int f0(int);
  }
  namespace N
  {
    using N2::f0;
  }

  template<class R,class B1>
  int
  f1(R(a)(B1));

  void f2()
  { f1(N::f0<int>); }
}
