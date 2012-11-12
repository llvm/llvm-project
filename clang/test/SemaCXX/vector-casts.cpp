// RUN: %clang_cc1 -fsyntax-only -verify %s
typedef int __v2si __attribute__((__vector_size__(8)));
typedef short __v4hi __attribute__((__vector_size__(8)));
typedef short __v8hi __attribute__((__vector_size__(16)));

struct S { }; // expected-note 2 {{candidate constructor}}

void f() {
  __v2si v2si;
  __v4hi v4hi;
  __v8hi v8hi;
  unsigned long long ll;
  unsigned char c;
  S s;
  
  (void)reinterpret_cast<__v2si>(v4hi);
  (void)(__v2si)v4hi;
  (void)reinterpret_cast<__v4hi>(v2si);
  (void)(__v4hi)v2si;
  (void)reinterpret_cast<unsigned long long>(v2si);
  (void)(unsigned long long)v2si;
  (void)reinterpret_cast<__v2si>(ll);
  (void)(__v2si)(ll);

  (void)reinterpret_cast<S>(v2si); // expected-error {{reinterpret_cast from '__v2si' to 'S' is not allowed}}
  (void)(S)v2si; // expected-error {{no matching conversion for C-style cast from '__v2si' to 'S'}}
  (void)reinterpret_cast<__v2si>(s); // expected-error {{reinterpret_cast from 'S' to '__v2si' is not allowed}}
  (void)(__v2si)s; // expected-error {{cannot convert 'S' to '__v2si' without a conversion operator}}
  
  (void)reinterpret_cast<unsigned char>(v2si); // expected-error {{reinterpret_cast from vector '__v2si' to scalar 'unsigned char' of different size}}
  (void)(unsigned char)v2si; // expected-error {{C-style cast from vector '__v2si' to scalar 'unsigned char' of different size}}
  (void)reinterpret_cast<__v2si>(c); // expected-error {{reinterpret_cast from scalar 'unsigned char' to vector '__v2si' of different size}}

  (void)reinterpret_cast<__v8hi>(v4hi); // expected-error {{reinterpret_cast from vector '__v4hi' to vector '__v8hi' of different size}}
  (void)(__v8hi)v4hi; // expected-error {{C-style cast from vector '__v4hi' to vector '__v8hi' of different size}}
  (void)reinterpret_cast<__v4hi>(v8hi); // expected-error {{reinterpret_cast from vector '__v8hi' to vector '__v4hi' of different size}}
  (void)(__v4hi)v8hi; // expected-error {{C-style cast from vector '__v8hi' to vector '__v4hi' of different size}}
}


