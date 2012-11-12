// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A { 
  int a;  // expected-note 4{{member found by ambiguous name lookup}}
  static int b;
  static int c; // expected-note 2{{member found by ambiguous name lookup}}

  enum E { enumerator };

  typedef int type;

  static void f(int);
  void f(float); // expected-note 2{{member found by ambiguous name lookup}}

  static void static_f(int);
  static void static_f(double);
};

struct B : A {
  int d; // expected-note 2{{member found by ambiguous name lookup}}

  enum E2 { enumerator2 };

  enum E3 { enumerator3 }; // expected-note 2{{member found by ambiguous name lookup}}
};

struct C : A {
  int c; // expected-note 2{{member found by ambiguous name lookup}}
  int d; // expected-note 2{{member found by ambiguous name lookup}}

  enum E3 { enumerator3_2 }; // expected-note 2{{member found by ambiguous name lookup}}
};

struct D : B, C {
  void test_lookup();
};

void test_lookup(D d) {
  d.a; // expected-error{{non-static member 'a' found in multiple base-class subobjects of type 'A':}}
  (void)d.b; // okay
  d.c; // expected-error{{member 'c' found in multiple base classes of different types}}
  d.d; // expected-error{{member 'd' found in multiple base classes of different types}}
  d.f(0); // expected-error{{non-static member 'f' found in multiple base-class subobjects of type 'A':}}
  d.static_f(0); // okay

  D::E e = D::enumerator; // okay
  D::type t = 0; // okay

  D::E2 e2 = D::enumerator2; // okay

  D::E3 e3; // expected-error{{multiple base classes}}
}

void D::test_lookup() {
  a; // expected-error{{non-static member 'a' found in multiple base-class subobjects of type 'A':}}
  (void)b; // okay
  c; // expected-error{{member 'c' found in multiple base classes of different types}}
  d; // expected-error{{member 'd' found in multiple base classes of different types}}
  f(0); // expected-error{{non-static member 'f' found in multiple base-class subobjects of type 'A':}}
  static_f(0); // okay

  E e = enumerator; // okay
  type t = 0; // okay

  E2 e2 = enumerator2; // okay

  E3 e3; // expected-error{{member 'E3' found in multiple base classes of different types}}
}

struct B2 : virtual A {
  int d; // expected-note 2{{member found by ambiguous name lookup}}

  enum E2 { enumerator2 };

  enum E3 { enumerator3 }; // expected-note 2 {{member found by ambiguous name lookup}}
};

struct C2 : virtual A {
  int c;
  int d; // expected-note 2{{member found by ambiguous name lookup}}

  enum E3 { enumerator3_2 }; // expected-note 2{{member found by ambiguous name lookup}}
};

struct D2 : B2, C2 { 
  void test_virtual_lookup();
};

struct F : A { };
struct G : F, D2 { 
  void test_virtual_lookup();
};

void test_virtual_lookup(D2 d2, G g) {
  (void)d2.a;
  (void)d2.b;
  (void)d2.c; // okay
  d2.d; // expected-error{{member 'd' found in multiple base classes of different types}}
  d2.f(0); // okay
  d2.static_f(0); // okay

  D2::E e = D2::enumerator; // okay
  D2::type t = 0; // okay

  D2::E2 e2 = D2::enumerator2; // okay

  D2::E3 e3; // expected-error{{member 'E3' found in multiple base classes of different types}}

  g.a; // expected-error{{non-static member 'a' found in multiple base-class subobjects of type 'A':}}
  g.static_f(0); // okay
}

void D2::test_virtual_lookup() {
  (void)a;
  (void)b;
  (void)c; // okay
  d; // expected-error{{member 'd' found in multiple base classes of different types}}
  f(0); // okay
  static_f(0); // okay

  E e = enumerator; // okay
  type t = 0; // okay

  E2 e2 = enumerator2; // okay

  E3 e3; // expected-error{{member 'E3' found in multiple base classes of different types}}
}

void G::test_virtual_lookup() {
  a; // expected-error{{non-static member 'a' found in multiple base-class subobjects of type 'A':}}
  static_f(0); // okay
}


struct HasMemberType1 {
  struct type { }; // expected-note{{member found by ambiguous name lookup}}
};

struct HasMemberType2 {
  struct type { }; // expected-note{{member found by ambiguous name lookup}}
};

struct HasAnotherMemberType : HasMemberType1, HasMemberType2 { 
  struct type { };
};

struct UsesAmbigMemberType : HasMemberType1, HasMemberType2 {
  type t; // expected-error{{member 'type' found in multiple base classes of different types}}
};

struct X0 {
  struct Inner {
    static const int m;
  };
  
  static const int n = 17;
};

const int X0::Inner::m = n;
