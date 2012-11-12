// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR8640 {
  template<class T1> struct C1 {
    virtual void c1() {
      T1 t1 = 3;  // expected-error {{cannot initialize a variable}}
    }
  };

  template<class T2> struct C2 {
    void c2() {
      new C1<T2>();  // expected-note {{in instantiation of member function}}
    }
  };

  void f() {
    C2<int*> c2;
    c2.c2();  // expected-note {{in instantiation of member function}}
  }
}

namespace PR9325 {
  template<typename T>
  class Target
  {
  public:
    virtual T Value() const
    {
      return 1; // expected-error{{cannot initialize return object of type 'int *' with an rvalue of type 'int'}}
    }
  };

  template<typename T>
  struct Provider
  {
    static Target<T> Instance;
  };

  template<typename T>
  Target<T> Provider<T>::Instance; // expected-note{{in instantiation of}}

  void f()
  {
    Target<int*>* traits = &Provider<int*>::Instance;
  }
}

namespace PR10020 {
  struct MG {
    virtual void Accept(int) = 0;
  };

  template <typename Type>
  struct GMG : MG {
    void Accept(int i) {
      static_cast<Type *>(0)->Accept(i); // expected-error{{member reference base}}
    }
    static GMG* Method() { return &singleton; } // expected-note{{in instantiation of}}
    static GMG singleton;
  };

  template <typename Type>
  GMG<Type> GMG<Type>::singleton;

  void test(void) {
    GMG<int>::Method(); // expected-note{{in instantiation of}}
  }
}
