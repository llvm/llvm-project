// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown -fsyntax-only -verify %s
typedef int __v2si __attribute__((__vector_size__(8)));
typedef short __v4hi __attribute__((__vector_size__(8)));
typedef short __v8hi __attribute__((__vector_size__(16)));
typedef short __v3hi __attribute__((__ext_vector_type__(3)));

struct S { }; // expected-note 3 {{candidate constructor}}

enum E : long long { Evalue };

void f() {
  __v2si v2si;
  __v3hi v3hi;
  __v4hi v4hi;
  __v8hi v8hi;
  unsigned long long ll;
  unsigned char c;
  S s;
  E e;
  
  (void)reinterpret_cast<__v2si>(v4hi);
  (void)(__v2si)v4hi;
  (void)reinterpret_cast<__v4hi>(v2si);
  (void)(__v4hi)v2si;
  (void)reinterpret_cast<unsigned long long>(v2si);
  (void)(unsigned long long)v2si;
  (void)reinterpret_cast<__v2si>(ll);
  (void)(__v2si)(ll);

  (void)(E)v2si; // expected-error {{C-style cast from '__v2si' (vector of 2 'int' values) to 'E' is not allowed}}
  (void)(__v2si)e; // expected-error {{C-style cast from 'E' to '__v2si' (vector of 2 'int' values)}}
  (void)reinterpret_cast<E>(v2si); // expected-error {{reinterpret_cast from '__v2si' (vector of 2 'int' values) to 'E' is not allowed}}
  (void)reinterpret_cast<__v2si>(e); // expected-error {{reinterpret_cast from 'E' to '__v2si' (vector of 2 'int' values)}}

  (void)reinterpret_cast<S>(v2si); // expected-error {{reinterpret_cast from '__v2si' (vector of 2 'int' values) to 'S' is not allowed}}
  (void)(S)v2si; // expected-error {{no matching conversion for C-style cast from '__v2si' (vector of 2 'int' values) to 'S'}}
  (void)reinterpret_cast<__v2si>(s); // expected-error {{reinterpret_cast from 'S' to '__v2si' (vector of 2 'int' values) is not allowed}}
  (void)(__v2si)s; // expected-error {{cannot convert 'S' to '__v2si' (vector of 2 'int' values) without a conversion operator}}
  
  (void)reinterpret_cast<unsigned char>(v2si); // expected-error {{reinterpret_cast from vector '__v2si' (vector of 2 'int' values) to scalar 'unsigned char' of different size}}
  (void)(unsigned char)v2si; // expected-error {{C-style cast from vector '__v2si' (vector of 2 'int' values) to scalar 'unsigned char' of different size}}
  (void)reinterpret_cast<__v2si>(c); // expected-error {{reinterpret_cast from scalar 'unsigned char' to vector '__v2si' (vector of 2 'int' values) of different size}}

  (void)reinterpret_cast<__v8hi>(v4hi); // expected-error {{reinterpret_cast from vector '__v4hi' (vector of 4 'short' values) to vector '__v8hi' (vector of 8 'short' values) of different size}}
  (void)(__v8hi)v4hi; // expected-error {{C-style cast from vector '__v4hi' (vector of 4 'short' values) to vector '__v8hi' (vector of 8 'short' values) of different size}}
  (void)reinterpret_cast<__v4hi>(v8hi); // expected-error {{reinterpret_cast from vector '__v8hi' (vector of 8 'short' values) to vector '__v4hi' (vector of 4 'short' values) of different size}}
  (void)(__v4hi)v8hi; // expected-error {{C-style cast from vector '__v8hi' (vector of 8 'short' values) to vector '__v4hi' (vector of 4 'short' values) of different size}}

  (void)(__v3hi)v4hi; // expected-error {{C-style cast from vector '__v4hi' (vector of 4 'short' values) to vector '__v3hi' (vector of 3 'short' values) of different size}}
  (void)(__v3hi)v2si; // expected-error {{C-style cast from vector '__v2si' (vector of 2 'int' values) to vector '__v3hi' (vector of 3 'short' values) of different size}}
  (void)(__v4hi)v3hi; // expected-error {{C-style cast from vector '__v3hi' (vector of 3 'short' values) to vector '__v4hi' (vector of 4 'short' values) of different size}}
  (void)(__v2si)v3hi; // expected-error {{C-style cast from vector '__v3hi' (vector of 3 'short' values) to vector '__v2si' (vector of 2 'int' values) of different size}}
  (void)reinterpret_cast<__v3hi>(v4hi); // expected-error {{reinterpret_cast from vector '__v4hi' (vector of 4 'short' values) to vector '__v3hi' (vector of 3 'short' values) of different size}}
  (void)reinterpret_cast<__v3hi>(v2si); // expected-error {{reinterpret_cast from vector '__v2si' (vector of 2 'int' values) to vector '__v3hi' (vector of 3 'short' values) of different size}}
  (void)reinterpret_cast<__v4hi>(v3hi); // expected-error {{reinterpret_cast from vector '__v3hi' (vector of 3 'short' values) to vector '__v4hi' (vector of 4 'short' values) of different size}}
  (void)reinterpret_cast<__v2si>(v3hi); // expected-error {{reinterpret_cast from vector '__v3hi' (vector of 3 'short' values) to vector '__v2si' (vector of 2 'int' values) of different size}}
}

struct testvec {
  __v2si v;
  void madd(const testvec& rhs) {
    v = v + rhs; // expected-error {{cannot convert between vector and non-scalar values}}
  }
  void madd2(testvec rhs) {
    v = v + rhs; // expected-error {{cannot convert between vector and non-scalar values}}
  }
};

// rdar://15931426
//   Conversions for return values.
__v4hi threeToFour(__v3hi v) { // expected-note {{not viable}}
  return v; // expected-error {{cannot initialize return object}}
}
__v3hi fourToThree(__v4hi v) { // expected-note {{not viable}}
  return v; // expected-error {{cannot initialize return object}}
}
//   Conversions for calls.
void call3to4(__v4hi v) {
  (void) threeToFour(v); // expected-error {{no matching function for call}}
}
void call4to3(__v3hi v) {
  (void) fourToThree(v); // expected-error {{no matching function for call}}
}
