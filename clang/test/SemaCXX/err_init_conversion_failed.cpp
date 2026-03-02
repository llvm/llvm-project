// RUN: %clang_cc1 -fsyntax-only -verify %s

void test0() {
  char variable = (void)0;
  // expected-error@-1{{cannot initialize a variable}}
}

void test1(int x = (void)0) {}
  // expected-error@-1{{cannot initialize a parameter}}
  // expected-note@-2{{here}}

int test2() {
  return (void)0;
  // expected-error@-1{{cannot initialize return object}}
}

struct S4 {
  S4() : x((void)0) {};
  // expected-error@-1{{cannot initialize a member subobject}}
  int x;
};

void test5() {
  int foo[2] = {1, (void)0};
  // expected-error@-1{{cannot initialize an array element}}
}

void test6() {
  new int((void)0);
  // expected-error@-1{{cannot initialize a new value}}
}

typedef short short2 __attribute__ ((__vector_size__ (2)));
void test10() {
  short2 V = { (void)0 };
  // expected-error@-1{{cannot initialize a vector element}}
}

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float4 __attribute__((ext_vector_type(4)));

void test14(const float2 in, const float2 out) {
  const float4 V = (float4){ in, out };
  // expected-error@-1{{cannot initialize a compound literal initializer}}
}

namespace template_test {
class S {
public:
   void foo(int);
};

template <class P> struct S2 {
  void (P::*a)(const int &);
};

void test_15() {
  S2<S> X = {&S::foo};
  // expected-error-re@-1{{cannot initialize a member subobject of type 'void (template_test::S::*)(const int &){{( __attribute__\(\(thiscall\)\))?}}' with an rvalue of type 'void (S::*)(int){{( __attribute__\(\(thiscall\)\))?}}': type mismatch at 1st parameter ('const int &' vs 'int')}}
}
}
