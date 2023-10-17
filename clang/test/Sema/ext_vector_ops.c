// RUN: %clang_cc1 %s -verify -fsyntax-only -Wvector-conversion -triple x86_64-apple-darwin10

typedef unsigned int v2u __attribute__ ((ext_vector_type(2)));
typedef int v2s __attribute__ ((ext_vector_type(2)));
typedef float v2f __attribute__ ((ext_vector_type(2)));

void test1(v2u v2ua, v2s v2sa, v2f v2fa) {
  // Bitwise binary operators
  (void)(v2ua & v2ua);
  (void)(v2fa & v2fa); // expected-error{{invalid operands to binary expression}}

  // Unary operators
  (void)(~v2ua);
  (void)(~v2fa); // expected-error{{invalid argument type 'v2f' (vector of 2 'float' values) to unary}}

  // Comparison operators
  v2sa = (v2ua==v2sa);

  // Arrays
  int array1[v2ua]; // expected-error{{size of array has non-integer type 'v2u' (vector of 2 'unsigned int' values}}
  int array2[17];
  // FIXME: error message below needs type!
  (void)(array2[v2ua]); // expected-error{{array subscript is not an integer}}

  v2u *v2u_ptr = 0;
  v2s *v2s_ptr;
}

void test_int_vector_scalar(unsigned int ua, v2u v2ua) {
  // Operations with one integer vector and one scalar. These splat the scalar.
  (void)(v2ua + ua);
  (void)(ua + v2ua);
  (void)(v2ua - ua);
  (void)(ua - v2ua);
  (void)(v2ua * ua);
  (void)(ua * v2ua);
  (void)(v2ua / ua);
  (void)(ua / v2ua);
  (void)(v2ua % ua);
  (void)(ua % v2ua);

  (void)(v2ua == ua);
  (void)(ua == v2ua);
  (void)(v2ua != ua);
  (void)(ua != v2ua);
  (void)(v2ua <= ua);
  (void)(ua <= v2ua);
  (void)(v2ua >= ua);
  (void)(ua >= v2ua);
  (void)(v2ua < ua);
  (void)(ua < v2ua);
  (void)(v2ua > ua);
  (void)(ua > v2ua);
  (void)(v2ua && ua);
  (void)(ua && v2ua);
  (void)(v2ua || ua);
  (void)(ua || v2ua);

  (void)(v2ua & ua);
  (void)(ua & v2ua);
  (void)(v2ua | ua);
  (void)(ua | v2ua);
  (void)(v2ua ^ ua);
  (void)(ua ^ v2ua);
  (void)(v2ua << ua);
  (void)(ua << v2ua);
  (void)(v2ua >> ua);
  (void)(ua >> v2ua);

  v2ua += ua;
  v2ua -= ua;
  v2ua *= ua;
  v2ua /= ua;
  v2ua %= ua;
  v2ua &= ua;
  v2ua |= ua;
  v2ua ^= ua;
  v2ua >>= ua;
  v2ua <<= ua;

  ua += v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
  ua -= v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
  ua *= v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
  ua /= v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
  ua %= v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
  ua &= v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
  ua |= v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
  ua ^= v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
  ua >>= v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
  ua <<= v2ua; // expected-error{{assigning to 'unsigned int' from incompatible type 'v2u'}}
}

void test_float_vector_scalar(float fa, unsigned int ua, v2f v2fa) {
  // Operations with one float vector and one scalar. These splat the scalar.
  (void)(v2fa + fa);
  (void)(fa + v2fa);
  (void)(v2fa - fa);
  (void)(fa - v2fa);
  (void)(v2fa * fa);
  (void)(fa * v2fa);
  (void)(v2fa / fa);
  (void)(fa / v2fa);
  (void)(v2fa % fa); // expected-error{{invalid operands to binary expression}}
  (void)(fa % v2fa); // expected-error{{invalid operands to binary expression}}

  (void)(v2fa == fa);
  (void)(fa == v2fa);
  (void)(v2fa != fa);
  (void)(fa != v2fa);
  (void)(v2fa <= fa);
  (void)(fa <= v2fa);
  (void)(v2fa >= fa);
  (void)(fa >= v2fa);
  (void)(v2fa < fa);
  (void)(fa < v2fa);
  (void)(v2fa > fa);
  (void)(fa > v2fa);
  (void)(v2fa && fa);
  (void)(fa && v2fa);
  (void)(v2fa || fa);
  (void)(fa || v2fa);

  (void)(v2fa & fa); // expected-error{{invalid operands to binary expression}}
  (void)(fa & v2fa); // expected-error{{invalid operands to binary expression}}
  (void)(v2fa | fa); // expected-error{{invalid operands to binary expression}}
  (void)(fa | v2fa); // expected-error{{invalid operands to binary expression}}
  (void)(v2fa ^ fa); // expected-error{{invalid operands to binary expression}}
  (void)(fa ^ v2fa); // expected-error{{invalid operands to binary expression}}
  (void)(v2fa << fa); // expected-error{{used type 'v2f' (vector of 2 'float' values) where integer is required}}
  (void)(v2fa << ua); // expected-error{{used type 'v2f' (vector of 2 'float' values) where integer is required}}
  (void)(fa << v2fa); // expected-error{{used type 'float' where integer is required}}
  (void)(ua << v2fa); // expected-error{{used type 'v2f' (vector of 2 'float' values) where integer is required}}
  (void)(v2fa >> fa); // expected-error{{used type 'v2f' (vector of 2 'float' values) where integer is required}}
  (void)(v2fa >> ua); // expected-error{{used type 'v2f' (vector of 2 'float' values) where integer is required}}
  (void)(fa >> v2fa); // expected-error{{used type 'float' where integer is required}}
  (void)(ua >> v2fa); // expected-error{{used type 'v2f' (vector of 2 'float' values) where integer is required}}

  v2fa += fa;
  v2fa -= fa;
  v2fa *= fa;
  v2fa /= fa;
  v2fa %= fa; // expected-error{{invalid operands to binary expression}}
  v2fa &= fa; // expected-error{{invalid operands to binary expression}}
  v2fa |= fa; // expected-error{{invalid operands to binary expression}}
  v2fa ^= fa; // expected-error{{invalid operands to binary expression}}
  v2fa >>= fa; // expected-error{{used type 'v2f' (vector of 2 'float' values) where integer is required}}
  v2fa <<= fa; // expected-error{{used type 'v2f' (vector of 2 'float' values) where integer is required}}

  fa += v2fa; // expected-error{{assigning to 'float' from incompatible type 'v2f'}}
  fa -= v2fa; // expected-error{{assigning to 'float' from incompatible type 'v2f'}}
  fa *= v2fa; // expected-error{{assigning to 'float' from incompatible type 'v2f'}}
  fa /= v2fa; // expected-error{{assigning to 'float' from incompatible type 'v2f'}}
  fa %= v2fa; // expected-error{{invalid operands to binary expression}}
  fa &= v2fa; // expected-error{{invalid operands to binary expression}}
  fa |= v2fa; // expected-error{{invalid operands to binary expression}}
  fa ^= v2fa; // expected-error{{invalid operands to binary expression}}
  fa >>= v2fa; // expected-error{{used type 'float' where integer is required}}
  fa <<= v2fa; // expected-error{{used type 'float' where integer is required}}
}

enum Enum { ENUM };

void test_enum_vector_scalar(enum Enum ea, v2u v2ua) {
  // Operations with one integer vector and one enum scalar.
  // These splat the scalar and do implicit integral conversions.
  (void)(v2ua + ea);
  (void)(ea + v2ua);
  (void)(v2ua - ea);
  (void)(ea - v2ua);
  (void)(v2ua * ea);
  (void)(ea * v2ua);
  (void)(v2ua / ea);
  (void)(ea / v2ua);
  (void)(v2ua % ea);
  (void)(ea % v2ua);

  (void)(v2ua == ea);
  (void)(ea == v2ua);
  (void)(v2ua != ea);
  (void)(ea != v2ua);
  (void)(v2ua <= ea);
  (void)(ea <= v2ua);
  (void)(v2ua >= ea);
  (void)(ea >= v2ua);
  (void)(v2ua < ea);
  (void)(ea < v2ua);
  (void)(v2ua > ea);
  (void)(ea > v2ua);
  (void)(v2ua && ea);
  (void)(ea && v2ua);
  (void)(v2ua || ea);
  (void)(ea || v2ua);

  (void)(v2ua & ea);
  (void)(ea & v2ua);
  (void)(v2ua | ea);
  (void)(ea | v2ua);
  (void)(v2ua ^ ea);
  (void)(ea ^ v2ua);
  (void)(v2ua << ea);
  (void)(ea << v2ua);
  (void)(v2ua >> ea);
  (void)(ea >> v2ua);

  v2ua += ea;
  v2ua -= ea;
  v2ua *= ea;
  v2ua /= ea;
  v2ua %= ea;
  v2ua &= ea;
  v2ua |= ea;
  v2ua ^= ea;
  v2ua >>= ea;
  v2ua <<= ea;

  ea += v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
  ea -= v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
  ea *= v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
  ea /= v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
  ea %= v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
  ea &= v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
  ea |= v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
  ea ^= v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
  ea >>= v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
  ea <<= v2ua; // expected-error{{assigning to 'enum Enum' from incompatible type 'v2u'}}
}


// An incomplete enum type doesn't count as an integral type.
enum Enum2;

void test_incomplete_enum(enum Enum2 *ea, v2u v2ua) {
  (void)(v2ua + *ea); // expected-error{{cannot convert between vector and non-scalar values}}
  (void)(*ea + v2ua); // expected-error{{cannot convert between vector and non-scalar values}}
}
