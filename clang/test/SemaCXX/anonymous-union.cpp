// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
struct X {
  union {
    float f3;
    double d2;
  } named;

  union {
    int i;
    float f;
    
    union { // expected-warning{{anonymous types declared in an anonymous union are an extension}}
      float f2;
      mutable double d;
    };
  };

  void test_unqual_references();

  struct { // expected-warning{{anonymous structs are a GNU extension}}
    int a;
    float b;
  };

  void test_unqual_references_const() const;

  mutable union { // expected-error{{anonymous union at class scope must not have a storage specifier}}
    float c1;
    double c2;
  };
};

void X::test_unqual_references() {
  i = 0;
  f = 0.0;
  f2 = f;
  d = f;
  f3 = 0; // expected-error{{use of undeclared identifier 'f3'}}
  a = 0;
}

void X::test_unqual_references_const() const { // expected-note 2{{member function 'X::test_unqual_references_const' is declared const here}}
  d = 0.0;
  f2 = 0; // expected-error{{cannot assign to non-static data member within const member function 'test_unqual_references_const'}}
  a = 0; // expected-error{{cannot assign to non-static data member within const member function 'test_unqual_references_const'}}
}

void test_unqual_references(X x, const X xc) {
  // expected-note@-1 2{{variable 'xc' declared const here}}
  x.i = 0;
  x.f = 0.0;
  x.f2 = x.f;
  x.d = x.f;
  x.f3 = 0; // expected-error{{no member named 'f3'}}
  x.a = 0;

  xc.d = 0.0;
  xc.f = 0; // expected-error{{cannot assign to variable 'xc' with const-qualified type 'const X'}}
  xc.a = 0; // expected-error{{cannot assign to variable 'xc' with const-qualified type 'const X'}}
}


struct Redecl {
  int x; // expected-note{{previous declaration is here}}
  class y { }; // expected-note{{previous declaration is here}}

  union {
    int x; // expected-error{{member of anonymous union redeclares 'x'}}
    float y; // expected-error{{member of anonymous union redeclares 'y'}}
    double z; // expected-note{{previous declaration is here}}
    double zz; // expected-note{{previous definition is here}}
  };

  int z; // expected-error{{duplicate member 'z'}}
  void zz(); // expected-error{{redefinition of 'zz' as different kind of symbol}}
};

union { // expected-error{{anonymous unions at namespace or global scope must be declared 'static'}}
  int int_val;
  float float_val;
};

extern "C++" {
union { int extern_cxx; }; // expected-error{{anonymous unions at namespace or global scope must be declared 'static'}}
}

static union {
  int int_val2; // expected-note{{previous definition is here}}
  float float_val2;
};

void PR21858() {
  void int_val2(); // expected-error{{redefinition of 'int_val2' as different kind of symbol}}
}

void f() {
  int_val2 = 0;
  float_val2 = 0.0;
}

void g() {
  union {
    int i;
    float f2;
  };
  i = 0;
  f2 = 0.0;
}

struct BadMembers {
  union {
    struct X { }; // expected-error {{types cannot be declared in an anonymous union}}
    struct { int x; int y; } y; // expected-warning{{anonymous types declared in an anonymous union are an extension}}
    
    void f(); // expected-error{{functions cannot be declared in an anonymous union}}
  private: int x1; // expected-error{{anonymous union cannot contain a private data member}}
  protected: float x2; // expected-error{{anonymous union cannot contain a protected data member}}
  };
};

// <rdar://problem/6481130>
typedef union { }; // expected-warning{{typedef requires a name}}

// <rdar://problem/7562438>
typedef struct objc_module *Foo ;

typedef struct _s {
    union {
        int a;
        int Foo;
    };
} s, *ps;

// <rdar://problem/7987650>
namespace test4 {
  class A {
    struct { // expected-warning{{anonymous structs are a GNU extension}}
      int s0; // expected-note {{declared private here}}
      double s1; // expected-note {{declared private here}}
      union { // expected-warning{{anonymous types declared in an anonymous struct are an extension}}
        int su0; // expected-note {{declared private here}}
        double su1; // expected-note {{declared private here}}
      };
    };
    union {
      int u0; // expected-note {{declared private here}}
      double u1; // expected-note {{declared private here}}
      struct { // expected-warning{{anonymous structs are a GNU extension}} expected-warning{{anonymous types declared in an anonymous union are an extension}}
        int us0; // expected-note {{declared private here}}
        double us1; // expected-note {{declared private here}}
      };
    };
  };

  void test() {
    A a;
    (void) a.s0;  // expected-error {{private member}}
    (void) a.s1;  // expected-error {{private member}}
    (void) a.su0; // expected-error {{private member}}
    (void) a.su1; // expected-error {{private member}}
    (void) a.u0;  // expected-error {{private member}}
    (void) a.u1;  // expected-error {{private member}}
    (void) a.us0; // expected-error {{private member}}
    (void) a.us1; // expected-error {{private member}}
  }
}

typedef void *voidPtr;

void f2() {
    union { int **ctxPtr; void **voidPtr; };
}

void foo_PR6741() {
    union {
        char *m_a;
        int *m_b;
    };
 
    if(1) {
        union {
            char *m_a;
            int *m_b;
        };
    }
}

namespace PR8326 {
  template <class T>
  class Foo {
  public:
    Foo()
      : x(0)
      , y(1){
    }
  
  private:
    const union { // expected-warning{{anonymous union cannot be 'const'}}
      struct { // expected-warning{{anonymous structs are a GNU extension}} expected-warning{{declared in an anonymous union}}
        T x;
        T y;
      };
      T v[2];
    };
  };

  Foo<int> baz;
}

namespace PR16630 {
  struct A { union { int x; float y; }; }; // expected-note {{member is declared here}}
  struct B : private A { using A::x; } b; // expected-note {{private}}
  void foo () {
    b.x = 10;
    b.y = 0; // expected-error {{'y' is a private member of 'PR16630::A'}}
  }
}
