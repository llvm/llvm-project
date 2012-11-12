// RUN: %clang_cc1 -fsyntax-only -verify %s 
struct ConvToBool {
  operator bool() const;
};

struct ConvToInt {
  operator int();
};

struct ExplicitConvToBool {
  explicit operator bool(); // expected-warning{{explicit conversion functions are a C++11 extension}}
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
  explicit operator int&(); // expected-warning{{explicit conversion functions are a C++11 extension}}
};

void test_explicit_bool(ExplicitConvToBool ecb) {
  bool b1(ecb); // okay
  bool b2 = ecb; // expected-error{{no viable conversion from 'ExplicitConvToBool' to 'bool'}}
  accepts_bool(ecb); // expected-error{{no matching function for call to}}
}

void test_explicit_conv_to_ref(ExplicitConvToRef ecr) {
  int& i1 = ecr; // expected-error{{non-const lvalue reference to type 'int' cannot bind to a value of unrelated type 'ExplicitConvToRef'}}
  int& i2(ecr); // okay
}

struct A { };
struct B { };
struct C {
  explicit operator A&(); // expected-warning{{explicit conversion functions are a C++11 extension}}
  operator B&(); // expected-note{{candidate}}
};

void test_copy_init_conversions(C c) {
  A &a = c; // expected-error{{no viable conversion from 'C' to 'A'}}
  B &b = c; // okay
}
