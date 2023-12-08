// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

const int global = 5;  // expected-note{{variable 'global' declared const here}}
void test1() {
  global = 2;  // expected-error{{cannot assign to variable 'global' with const-qualified type 'const int'}}
}

void test2 () {
  const int local = 5;  // expected-note{{variable 'local' declared const here}}
  local = 0;  // expected-error{{cannot assign to variable 'local' with const-qualified type 'const int'}}
}

void test2 (const int parameter) {  // expected-note{{variable 'parameter' declared const here}}
  parameter = 2;  // expected-error{{cannot assign to variable 'parameter' with const-qualified type 'const int'}}
}

class test3 {
  int field;
  const int const_field = 1;  // expected-note 2{{non-static data member 'const_field' declared const here}}
  static const int static_const_field = 1;  // expected-note 2{{variable 'static_const_field' declared const here}}
  void test() {
    const_field = 4;  // expected-error{{cannot assign to non-static data member 'const_field' with const-qualified type 'const int'}}
    static_const_field = 4;  // expected-error{{cannot assign to variable 'static_const_field' with const-qualified type 'const int'}}
  }
  void test_const() const { // expected-note 2{{member function 'test3::test_const' is declared const here}}
    field = 4;  // expected-error{{cannot assign to non-static data member within const member function 'test_const'}}
    const_field = 4 ;  // expected-error{{cannot assign to non-static data member 'const_field' with const-qualified type 'const int'}}
    static_const_field = 4;  // expected-error{{cannot assign to variable 'static_const_field' with const-qualified type 'const int'}}
  }
};

const int &return_const_ref();  // expected-note{{function 'return_const_ref' which returns const-qualified type 'const int &' declared here}}

void test4() {
  return_const_ref() = 10;  // expected-error{{cannot assign to return value because function 'return_const_ref' returns a const value}}
}

struct S5 {
  int field;
  const int const_field = 4;  // expected-note {{non-static data member 'const_field' declared const here}}
};

void test5() {
  S5 s5;
  s5.field = 5;
  s5.const_field = 5;  // expected-error{{cannot assign to non-static data member 'const_field' with const-qualified type 'const int'}}
}

struct U1 {
  int a = 5;
};

struct U2 {
  U1 u1;
};

struct U3 {
  const U2 u2 = U2();  // expected-note{{non-static data member 'u2' declared const here}}
};

struct U4 {
  U3 u3;
};

void test6() {
  U4 u4;
  u4.u3.u2.u1.a = 5;  // expected-error{{cannot assign to non-static data member 'u2' with const-qualified type 'const U2'}}
}

struct A {
  int z;
};
struct B {
  A a;
};
struct C {
  B b;
  C();
};
const C &getc(); // expected-note{{function 'getc' which returns const-qualified type 'const C &' declared here}}
void test7() {
  const C c;    // expected-note{{variable 'c' declared const here}}
  c.b.a.z = 5;  // expected-error{{cannot assign to variable 'c' with const-qualified type 'const C'}}

  getc().b.a.z = 5;  // expected-error{{cannot assign to return value because function 'getc' returns a const value}}
}

struct D { const int n; };  // expected-note 2{{non-static data member 'n' declared const here}}
struct E { D *const d = 0; };
void test8() {
  extern D *const d;
  d->n = 0;  // expected-error{{cannot assign to non-static data member 'n' with const-qualified type 'const int'}}

  E e;
  e.d->n = 0;  // expected-error{{cannot assign to non-static data member 'n' with const-qualified type 'const int'}}
}

struct F { int n; };
struct G { const F *f; };  // expected-note{{non-static data member 'f' declared const here}}
void test10() {
  const F *f;  // expected-note{{variable 'f' declared const here}}
  f->n = 0;    // expected-error{{cannot assign to variable 'f' with const-qualified type 'const F *'}}

  G g;
  g.f->n = 0;  // expected-error{{cannot assign to non-static data member 'f' with const-qualified type 'const F *'}}
}

void test11(
    const int x,  // expected-note{{variable 'x' declared const here}}
    const int& y  // expected-note{{variable 'y' declared const here}}
    ) {
  x = 5;  // expected-error{{cannot assign to variable 'x' with const-qualified type 'const int'}}
  y = 5;  // expected-error{{cannot assign to variable 'y' with const-qualified type 'const int &'}}
}

struct H {
  const int a = 0;   // expected-note{{non-static data member 'a' declared const here}}
  const int &b = a;  // expected-note{{non-static data member 'b' declared const here}}
};

void test12(H h) {
  h.a = 1;  // expected-error {{cannot assign to non-static data member 'a' with const-qualified type 'const int'}}
  h.b = 2;  // expected-error {{cannot assign to non-static data member 'b' with const-qualified type 'const int &'}}
}

void test() {
  typedef const int &Func();

  Func &bar();
  bar()() = 0; // expected-error {{read-only variable is not assignable}}
}

typedef float float4 __attribute__((ext_vector_type(4)));
struct OhNo {
  float4 v;
  void AssignMe() const { v.x = 1; } // expected-error {{cannot assign to non-static data member within const member function 'AssignMe'}} \
                                        expected-note {{member function 'OhNo::AssignMe' is declared const here}}
};

typedef float float4_2 __attribute__((__vector_size__(16)));
struct OhNo2 {
  float4_2 v;
  void AssignMe() const { v[0] = 1; } // expected-error {{cannot assign to non-static data member within const member function 'AssignMe'}} \
                                        expected-note {{member function 'OhNo2::AssignMe' is declared const here}}
};

struct OhNo3 {
  float v[4];
  void AssignMe() const { v[0] = 1; } // expected-error {{cannot assign to non-static data member within const member function 'AssignMe'}} \
                                        expected-note {{member function 'OhNo3::AssignMe' is declared const here}}
};
