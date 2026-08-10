// RUN: %clang_cc1 -fsyntax-only -verify %s 
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct ConvToBool {
  operator bool() const;
};

struct ConvToInt {
  operator int();
};

struct ExplicitConvToBool {
  explicit operator bool(); // expected-note {{explicit}}
#if __cplusplus <= 199711L // C++03 or earlier modes
  // expected-warning@-2{{explicit conversion functions are a C++11 extension}}
#endif
};

void test_conv_to_bool(ConvToBool ctb, ConvToInt cti, ExplicitConvToBool ecb) {
  if (ctb) { }
  if (cti) { }
  if (ecb) { }
  for (; ctb; ) { }
  for (; cti; ) { }
  for (; ecb; ) { }
  while (ctb) { };
  while (cti) { }
  while (ecb) { }
  do { } while (ctb);
  do { } while (cti);
  do { } while (ecb);

  if (!ctb) { }
  if (!cti) { }
  if (!ecb) { }

  bool b1 = !ecb;
  if (ctb && ecb) { }
  bool b2 = ctb && ecb;
  if (ctb || ecb) { }
  bool b3 = ctb || ecb;
}

void accepts_bool(bool) { } // expected-note{{candidate function}}

struct ExplicitConvToRef {
  explicit operator int&(); // expected-note {{explicit}}
#if (__cplusplus <= 199711L) // C++03 or earlier modes
  // expected-warning@-2{{explicit conversion functions are a C++11 extension}}
#endif
};

void test_explicit_bool(ExplicitConvToBool ecb) {
  bool b1(ecb); // okay
  bool b2 = ecb; // expected-error{{no viable conversion from 'ExplicitConvToBool' to 'bool'}}
  accepts_bool(ecb); // expected-error{{no matching function for call to}}
}

void test_explicit_conv_to_ref(ExplicitConvToRef ecr) {
  int& i1 = ecr; // expected-error{{no viable conversion from 'ExplicitConvToRef' to 'int'}}
  int& i2(ecr); // okay
}

struct A { };
struct B { };
struct C {
  explicit operator A&();  // expected-note {{explicit}}
#if __cplusplus <= 199711L // C++03 or earlier modes
// expected-warning@-2{{explicit conversion functions are a C++11 extension}}
#endif
  operator B&(); // expected-note{{candidate}}
};

void test_copy_init_conversions(C c) {
  A &a = c; // expected-error{{no viable conversion from 'C' to 'A'}}
  B &b = c; // okay
}
