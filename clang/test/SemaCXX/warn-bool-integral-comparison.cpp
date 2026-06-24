// RUN: %clang_cc1 -fsyntax-only -verify -Wbool-integral-comparison %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wextra %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,priority -Wbool-integral-comparison -Wtautological-unsigned-zero-compare %s
// RUN: %clang_cc1 -fsyntax-only -verify=wall -Wall %s

// wall-no-diagnostics

void integral_comparisons(bool b, char c, int i, unsigned u, bool other) {
  (void)(b == c); // expected-warning {{comparison between 'bool' and integral type 'char' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
  (void)(c != b); // expected-warning {{comparison between 'bool' and integral type 'char' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
  (void)(b < i);  // expected-warning {{comparison between 'bool' and integral type 'int' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
  (void)(u >= b); // expected-warning {{comparison between 'bool' and integral type 'unsigned int' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}

  (void)(b == other);
  (void)(b < other);
}

void reference_operands(bool b, bool &br, int i, int &ir) {
  (void)(br == i);  // expected-warning {{comparison between 'bool' and integral type 'int' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
  (void)(b == ir);  // expected-warning {{comparison between 'bool' and integral type 'int' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
  (void)(ir == br); // expected-warning {{comparison between 'bool' and integral type 'int' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
}

void parenthesized_operands(bool b, char c) {
  (void)((b) == (c)); // expected-warning {{comparison between 'bool' and integral type 'char' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
}

void qualified_operands(const bool b, volatile char c) {
  (void)(b == c); // expected-warning {{comparison between 'const bool' and integral type 'volatile char' is suspicious; the 'const bool' operand is converted to an integral value that can only be 0 or 1}}
}

typedef bool Bool;
using Int = int;

void alias_operands(Bool b, Int i) {
  (void)(b == i); // expected-warning {{comparison between 'Bool' (aka 'bool') and integral type 'Int' (aka 'int') is suspicious; the 'Bool' operand is converted to an integral value that can only be 0 or 1}}
}

void bool_like_integral_operands(bool b, bool other, int i, int j) {
  (void)(b == +other);
  (void)(b == +(i < j));
  (void)(b == (i == j));
  (void)(b == -other); // expected-warning {{comparison between 'bool' and integral type 'int' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
}

template <typename T> bool dependent_template_comparison(bool b, T t) {
  return b == t;
}

void instantiate_template_comparison(bool b, int i) {
  (void)dependent_template_comparison(b, i);
}

template <typename T> bool non_dependent_template_comparison(bool b, int i) {
  return b == i; // expected-warning {{comparison between 'bool' and integral type 'int' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
}

struct BitFields {
  unsigned one_bit : 1;
  unsigned two_bits : 2;
  int signed_one_bit : 1;
  bool bool_one_bit : 1;
};

void bit_field_comparisons(bool b, BitFields bf, BitFields *bfp) {
  (void)(b == bf.two_bits); // expected-warning {{comparison between 'bool' and integral type 'unsigned int' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
  (void)(bf.signed_one_bit == b); // expected-warning {{comparison between 'bool' and integral type 'int' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}

  (void)(b == bf.one_bit);
  (void)(bfp->one_bit == b);
  (void)(b == bf.bool_one_bit);
}

void bitint_comparisons(bool b, unsigned _BitInt(1) one_bit,
                        unsigned _BitInt(2) two_bits) {
  (void)(b == one_bit);
  (void)(two_bits == b); // expected-warning {{comparison between 'bool' and integral type 'unsigned _BitInt(2)' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
}

void constant_integral(bool b) {
  constexpr int one = 1;
  const int zero = 0;

  (void)(b == 1);
  (void)(0 != b);
  (void)(b < one);
  (void)(zero >= b);
  (void)(true >= 'a');
}

enum E { e0, e1 };

void enum_comparisons(bool b, E e) {
  (void)(b == e); // expected-warning {{comparison between 'bool' and enumeration type 'E' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
  (void)(e != b); // expected-warning {{comparison between 'bool' and enumeration type 'E' is suspicious; the 'bool' operand is converted to an integral value that can only be 0 or 1}}
}

void constant_enum(bool b) {
  constexpr E zero = e0;
  const E one = e1;

  (void)(b == e0);
  (void)(e1 != b);
  (void)(b == zero);
  (void)(one != b);
}

void tautological_compare_priority(unsigned u) {
  (void)(false <= u); // priority-warning {{comparison of 0 <= unsigned expression is always true}}
  (void)(u >= false); // priority-warning {{comparison of unsigned expression >= 0 is always true}}
}
