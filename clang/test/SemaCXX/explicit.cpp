// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
namespace Constructor {
struct A {
  A(int);
};

struct B {
  explicit B(int);
};

B::B(int) { }

struct C {
  void f(const A&);
  void f(const B&);
};

void f(C c) {
  c.f(10);
}
}

namespace Conversion {
  struct A {
    operator int();
    explicit operator bool();
  };

  A::operator bool() { return false; } 

  struct B {
    void f(int);
    void f(bool);
  };

  void f(A a, B b) {
    b.f(a);
  }
  
  void testExplicit()
  {
    // Taken from 12.3.2p2
    class Y { }; // expected-note {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'Z' to 'const Y &' for 1st argument}} \
					          expected-note {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'Z' to 'Y &&' for 1st argument}} \
                    expected-note {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'Z' to 'const Y &' for 1st argument}} \
          expected-note {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'Z' to 'Y &&' for 1st argument}}

    struct Z {
      explicit operator Y() const;
      explicit operator int() const;
    };
    
    Z z;
    // 13.3.1.4p1 & 8.5p16:
    Y y2 = z; // expected-error {{no viable conversion from 'Z' to 'Y'}}
    Y y3 = (Y)z;
    Y y4 = Y(z);
    Y y5 = static_cast<Y>(z);
    // 13.3.1.5p1 & 8.5p16:
    int i1 = (int)z;
    int i2 = int(z);
    int i3 = static_cast<int>(z);
    int i4(z);
    // 13.3.1.6p1 & 8.5.3p5:
    const Y& y6 = z; // expected-error {{no viable conversion from 'Z' to 'const Y'}}
    const int& y7(z);
  }
  
  void testBool() {
    struct Bool {
      operator bool();
    };

    struct NotBool {
      explicit operator bool(); // expected-note {{conversion to integral type 'bool'}}
    };
    Bool    b;
    NotBool n;

    (void) (1 + b);
    (void) (1 + n); // expected-error {{invalid operands to binary expression ('int' and 'NotBool')}}
    
    // 5.3.1p9:
    (void) (!b);
    (void) (!n);
    
    // 5.14p1:
    (void) (b && true);
    (void) (n && true);
    
    // 5.15p1:
    (void) (b || true);
    (void) (n || true);
    
    // 5.16p1:
    (void) (b ? 0 : 1);
    (void) (n ? 0: 1);
    
    // 5.19p5:
    // TODO: After constexpr has been implemented
    
    // 6.4p4:
    if (b) {}
    if (n) {}
    
    // 6.4.2p2:
    switch (b) {} // expected-warning {{switch condition has boolean value}}
    switch (n) {} // expected-error {{switch condition type 'NotBool' requires explicit conversion to 'bool'}} \
                     expected-warning {{switch condition has boolean value}}
    
    // 6.5.1:
    while (b) {}
    while (n) {}
    
    // 6.5.2p1:
    do {} while (b);
    do {} while (n);
    
    // 6.5.3:
    for (;b;) {}
    for (;n;) {}
  }
  
  void testNew()
  {
    // 5.3.4p6:
    struct Int {
      operator int();
    };
    struct NotInt {
      explicit operator int(); // expected-note {{conversion to integral type 'int' declared here}}
    };
    
    Int    i;
    NotInt ni;
    
    new int[i];
    new int[ni]; // expected-error {{array size expression of type 'NotInt' requires explicit conversion to type 'int'}}
  }
  
  void testDelete()
  {
    // 5.3.5pp2:
    struct Ptr {
      operator int*();
    };
    struct NotPtr {
      explicit operator int*();
    };
    
    Ptr    p;
    NotPtr np;
    
    delete p;
    delete np; // expected-error {{cannot delete expression of type 'NotPtr'}}
  }
  
  void testFunctionPointer()
  {
    // 13.3.1.1.2p2:
    using Func = void(*)(int);
    
    struct FP {
      operator Func();
    };
    struct NotFP {
      explicit operator Func();
    };
    
    FP    fp;
    NotFP nfp;
    fp(1);
    nfp(1); // expected-error {{type 'NotFP' does not provide a call operator}}
  }
}
