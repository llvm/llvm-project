// RUN: %clang_cc1 -triple x86_64-linux-pc -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++17
// Note that this test depends on the size of long-long to be different from
// int, so it specifies a triple.

using FourShorts = short __attribute__((__vector_size__(8)));
using TwoInts = int __attribute__((__vector_size__(8)));
using EightInts = int __attribute__((__vector_size__(32)));
using TwoUInts = unsigned __attribute__((__vector_size__(8)));
using FourInts = int __attribute__((__vector_size__(16)));
using FourUInts = unsigned __attribute__((__vector_size__(16)));
using TwoLongLong = long long __attribute__((__vector_size__(16)));
using FourLongLong = long long __attribute__((__vector_size__(32)));
using TwoFloats = float __attribute__((__vector_size__(8)));
using FourFloats = float __attribute__((__vector_size__(16)));
using TwoDoubles = double __attribute__((__vector_size__(16)));
using FourDoubles = double __attribute__((__vector_size__(32)));
using EightBools = bool __attribute__((ext_vector_type(8)));

EightInts eight_ints;
EightBools eight_bools;
EightBools other_eight_bools;
bool one_bool;

// Check the rules of the LHS/RHS of the conditional operator.
void Operations() {
  // Legal binary
  // (void)(eight_bools | other_eight_bools);
  // (void)(eight_bools & other_eight_bools);
  // (void)(eight_bools ^ other_eight_bools);
  // (void)(~eight_bools);
  // (void)(!eight_bools);

  // // Legal comparison
  // (void)(eight_bools == other_eight_bools);
  // (void)(eight_bools != other_eight_bools);
  // (void)(eight_bools < other_eight_bools);
  // (void)(eight_bools <= other_eight_bools);
  // (void)(eight_bools > other_eight_bools);
  // (void)(eight_bools >= other_eight_bools);

  // // Legal assignments
  // (void)(eight_bools |= other_eight_bools);
  // (void)(eight_bools &= other_eight_bools);
  // (void)(eight_bools ^= other_eight_bools);

  // Illegal operators
  (void)(eight_bools || other_eight_bools); // expected-error@47 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools && other_eight_bools); // expected-error@48 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools + other_eight_bools);  // expected-error@49 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools - other_eight_bools);  // expected-error@50 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools * other_eight_bools);  // expected-error@51 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools << other_eight_bools); // expected-error@52 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools >> other_eight_bools); // expected-error@53 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools / other_eight_bools);  // expected-error@54 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools % other_eight_bools);  // expected-error@55 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}

  // Illegal assignment
  (void)(eight_bools += other_eight_bools);  // expected-error@58 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools -= other_eight_bools);  // expected-error@59 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools *= other_eight_bools);  // expected-error@60 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools <<= other_eight_bools); // expected-error@61 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools >>= other_eight_bools); // expected-error@62 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools /= other_eight_bools);  // expected-error@63 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}
  (void)(eight_bools %= other_eight_bools);  // expected-error@64 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightBools')}}

  // Illegal in/decrements
  (void)(eight_bools++); // expected-error@67 {{cannot increment value of type 'EightBools' (vector of 8 'bool' values)}}
  (void)(++eight_bools); // expected-error@68 {{cannot increment value of type 'EightBools' (vector of 8 'bool' values)}}
  (void)(eight_bools--); // expected-error@69 {{cannot decrement value of type 'EightBools' (vector of 8 'bool' values)}}
  (void)(--eight_bools); // expected-error@70 {{cannot decrement value of type 'EightBools' (vector of 8 'bool' values)}}

  // No implicit promotion
  (void)(eight_bools + eight_ints); // expected-error@73 {{invalid operands to binary expression ('EightBools' (vector of 8 'bool' values) and 'EightInts' (vector of 8 'int' values))}}
  (void)(eight_ints - eight_bools); // expected-error@74 {{invalid operands to binary expression ('EightInts' (vector of 8 'int' values) and 'EightBools' (vector of 8 'bool' values))}}
}

// Allow scalar-to-vector broadcast. Do not allow bool vector conversions.
void Conversions() {
  (void)((long)eight_bools); // expected-error@79 {{C-style cast from vector 'EightBools' (vector of 8 'bool' values) to scalar 'long' of different size}}
  (void)((EightBools) one_bool); // Scalar-to-vector broadcast.
  (void)((char)eight_bools); // expected-error@81 {{C-style cast from vector 'EightBools' (vector of 8 'bool' values) to scalar 'char' of different size}}
}

void foo(const bool& X);

// Disallow element-wise access.
bool* ElementRefs() {
  eight_bools.y = false; // expected-error@88 {{illegal vector component name ''y''}}
  &eight_bools.z;        // expected-error@89 {{illegal vector component name ''z''}}
  foo(eight_bools.w);    // expected-error@90 {{illegal vector component name ''w''}}
  foo(eight_bools.wyx);  // expected-error@91 {{illegal vector component name ''wyx''}}
}

void Sizeof() {
  using FourBools = bool __attribute__((ext_vector_type(4)));
  using NineBools = bool __attribute__((ext_vector_type(9)));
  using TwentyEightBools = bool __attribute__((ext_vector_type(28)));
  using ThirtyThreeBools = bool __attribute__((ext_vector_type(33)));
  using SixtyFiveBools = bool __attribute__((ext_vector_type(65)));
  using Bool129 = bool __attribute__((ext_vector_type(129)));
  using Bool150 = bool __attribute__((ext_vector_type(150)));
  using Bool195 = bool __attribute__((ext_vector_type(195)));
  using Bool257 = bool __attribute__((ext_vector_type(257)));
  static_assert(sizeof(FourBools) == 1);
  static_assert(sizeof(EightBools) == 1);
  static_assert(sizeof(NineBools) == 2);
  static_assert(sizeof(TwentyEightBools) == 4);
  static_assert(sizeof(ThirtyThreeBools) == 8);
  static_assert(sizeof(SixtyFiveBools) == 16);
  static_assert(sizeof(Bool129) == 32);
  static_assert(sizeof(Bool150) == 32);
  static_assert(sizeof(Bool195) == 32);
  static_assert(sizeof(Bool257) == 64);
}
