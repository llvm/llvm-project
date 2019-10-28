// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  struct BASE { 
    operator int &(); // expected-note {{candidate function}}
  }; 
  struct BASE1 { 
    operator int &(); // expected-note {{candidate function}}
  }; 

  struct B : public BASE, BASE1 {};

  extern B f(); 
  B b1;

  void func(const int ci, const char cc);
  void func(const char ci, const B b); // expected-note {{candidate function}}
  void func(const B b, const int ci); // expected-note {{candidate function}}

  const int Test1() {

    func(b1, f()); // expected-error {{call to 'func' is ambiguous}}
    return f(); // expected-error {{conversion from 'test0::B' to 'const int' is ambiguous}}
  }

  // This used to crash when comparing the two operands.
  void func2(const char cc); // expected-note {{candidate function}}
  void func2(const int ci); // expected-note {{candidate function}}
  void Test2() {
    func2(b1); // expected-error {{call to 'func2' is ambiguous}}
  }
}

namespace test1 {
  struct E;
  struct A { 
    A (E&); 
  };

  struct E { 
    operator A (); 
  };

  struct C { 
    C (E&);  
  };

  void f1(A);	// expected-note {{candidate function}}
  void f1(C);	// expected-note {{candidate function}}

  void Test2()
  {
    E b;
    f1(b);  // expected-error {{call to 'f1' is ambiguous}}	
            // ambiguous because b -> C via constructor and
            // b -> A via constructor or conversion function.
  }
}

namespace rdar8876150 {
  struct A { operator bool(); };
  struct B : A { };
  struct C : A { };
  struct D : B, C { };

  bool f(D d) { return !d; } // expected-error{{ambiguous conversion from derived class 'rdar8876150::D' to base class 'rdar8876150::A':}}
}

namespace assignment {
  struct A { operator short(); operator bool(); }; // expected-note 2{{candidate}}
  void f(int n, A a) { n = a; } // expected-error{{ambiguous}}
}
