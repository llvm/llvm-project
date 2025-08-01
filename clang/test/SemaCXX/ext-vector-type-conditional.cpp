// RUN: %clang_cc1 -triple x86_64-linux-pc -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++17
// Note that this test depends on the size of long-long to be different from
// int, so it specifies a triple.

using FourShorts = short __attribute__((ext_vector_type(4)));
using TwoInts = int __attribute__((ext_vector_type(2)));
using TwoUInts = unsigned __attribute__((ext_vector_type(2)));
using FourInts = int __attribute__((ext_vector_type(4)));
using FourUInts = unsigned __attribute__((ext_vector_type(4)));
using TwoLongLong = long long __attribute__((ext_vector_type(2)));
using FourLongLong = long long __attribute__((ext_vector_type(4)));
using TwoFloats = float __attribute__((ext_vector_type(2)));
using FourFloats = float __attribute__((ext_vector_type(4)));
using TwoDoubles = double __attribute__((ext_vector_type(2)));
using FourDoubles = double __attribute__((ext_vector_type(4)));

FourShorts four_shorts;
TwoInts two_ints;
TwoUInts two_uints;
FourInts four_ints;
FourUInts four_uints;
TwoLongLong two_ll;
FourLongLong four_ll;
TwoFloats two_floats;
FourFloats four_floats;
TwoDoubles two_doubles;
FourDoubles four_doubles;

enum E {};
enum class SE {};
E e;
SE se;

// Check the rules of the condition of the conditional operator.
void Condition() {
  // Only int types are allowed here, the rest should fail to convert to bool.
  (void)(four_floats ? 1 : 1); // expected-error {{is not contextually convertible to 'bool'}}}
  (void)(two_doubles ? 1 : 1); // expected-error {{is not contextually convertible to 'bool'}}}
}

// Check the rules of the LHS/RHS of the conditional operator.
void Operands() {
  (void)(four_ints ? four_ints : throw 1); // expected-error {{GNU vector conditional operand cannot be a throw expression}}
  (void)(four_ints ? throw 1 : four_ints); // expected-error {{GNU vector conditional operand cannot be a throw expression}}
  (void)(four_ints ?: throw 1);            // expected-error {{GNU vector conditional operand cannot be a throw expression}}
  (void)(four_ints ? (void)1 : four_ints); // expected-error {{GNU vector conditional operand cannot be void}}
  (void)(four_ints ?: (void)1);            // expected-error {{GNU vector conditional operand cannot be void}}

  // Vector types must be the same element size as the condition.
  (void)(four_ints ? two_ll : two_ll);             // expected-error {{vector condition type 'FourInts' (vector of 4 'int' values) and result type 'TwoLongLong' (vector of 2 'long long' values) do not have the same number of elements}}
  (void)(four_ints ? four_ll : four_ll);           // expected-error {{vector condition type 'FourInts' (vector of 4 'int' values) and result type 'FourLongLong' (vector of 4 'long long' values) do not have elements of the same size}}
  (void)(four_ints ? two_doubles : two_doubles);   // expected-error {{vector condition type 'FourInts' (vector of 4 'int' values) and result type 'TwoDoubles' (vector of 2 'double' values) do not have the same number of elements}}
  (void)(four_ints ? four_doubles : four_doubles); // expected-error {{vector condition type 'FourInts' (vector of 4 'int' values) and result type 'FourDoubles' (vector of 4 'double' values) do not have elements of the same size}}
  (void)(four_ints ?: two_ints);                   // expected-error {{vector operands to the vector conditional must be the same type ('FourInts' (vector of 4 'int' values) and 'TwoInts' (vector of 2 'int' values)}}
  (void)(four_ints ?: four_doubles);               // expected-error {{vector operands to the vector conditional must be the same type ('FourInts' (vector of 4 'int' values) and 'FourDoubles' (vector of 4 'double' values)}}

  // Scalars are promoted, but must be the same element size.
  (void)(four_ints ? 3.0f : 3.0); // expected-error {{vector condition type 'FourInts' (vector of 4 'int' values) and result type 'double __attribute__((ext_vector_type(4)))' (vector of 4 'double' values) do not have elements of the same size}}
  (void)(four_ints ? 5ll : 5);    // expected-error {{vector condition type 'FourInts' (vector of 4 'int' values) and result type 'long long __attribute__((ext_vector_type(4)))' (vector of 4 'long long' values) do not have elements of the same size}}
  (void)(four_ints ?: 3.0);       // expected-error {{annot convert between vector values of different size ('FourInts' (vector of 4 'int' values) and 'double')}}
  (void)(four_ints ?: 5ll);       // We allow this despite GCc not allowing this since we support integral->vector-integral conversions despite integer rank.

  // This one would be allowed in GCC, but we don't allow vectors of enum. Also,
  // the error message isn't perfect, since it is only going to be a problem
  // when both sides are an enum, otherwise it'll be promoted to whatever type
  // the other side causes.
  (void)(four_ints ? e : e);                          // expected-error {{enumeration type 'E' is not allowed in a vector conditional}}
  (void)(four_ints ? se : se);                        // expected-error {{enumeration type 'SE' is not allowed in a vector conditional}}
  (void)(four_shorts ? (short)5 : (unsigned short)5); // expected-error {{vector condition type 'FourShorts' (vector of 4 'short' values) and result type 'int __attribute__((ext_vector_type(4)))' (vector of 4 'int' values) do not have elements of the same size}}

  // They must also be convertible.
  (void)(four_ints ? 3.0f : 5u);
  (void)(four_ints ? 3.0f : 5);
  unsigned us = 5u;
  int sint = 5;
  short shrt = 5;
  unsigned short uss = 5u;
  // The following 2 error in GCC for truncation errors, but it seems
  // unimportant and inconsistent to enforce that rule.
  (void)(four_ints ? 3.0f : us);
  (void)(four_ints ? 3.0f : sint);

  // Test promotion:
  (void)(four_shorts ? uss : shrt);  // expected-error {{vector condition type 'FourShorts' (vector of 4 'short' values) and result type 'int __attribute__((ext_vector_type(4)))' (vector of 4 'int' values) do not have elements of the same size}}
  (void)(four_shorts ? shrt : shrt); // should be fine.
  (void)(four_ints ? uss : shrt);    // should be fine, since they get promoted to int.
  (void)(four_ints ? shrt : shrt);   // expected-error {{vector condition type 'FourInts' (vector of 4 'int' values) and result type 'short __attribute__((ext_vector_type(4)))' (vector of 4 'short' values) do not have elements of the same size}}

  // Vectors must be the same type as each other.
  (void)(four_ints ? four_uints : four_floats); // expected-error {{vector operands to the vector conditional must be the same type ('FourUInts' (vector of 4 'unsigned int' values) and 'FourFloats' (vector of 4 'float' values))}}
  (void)(four_ints ? four_uints : four_ints);   // expected-error {{vector operands to the vector conditional must be the same type ('FourUInts' (vector of 4 'unsigned int' values) and 'FourInts' (vector of 4 'int' values))}}
  (void)(four_ints ? four_ints : four_uints);   // expected-error {{vector operands to the vector conditional must be the same type ('FourInts' (vector of 4 'int' values) and 'FourUInts' (vector of 4 'unsigned int' values))}}

  (void)(four_ints ? four_uints : 3.0f); // expected-error {{cannot convert between vector values of different size ('FourUInts' (vector of 4 'unsigned int' values) and 'float')}}
  (void)(four_ints ? four_ints : 3.0f);  // expected-error {{cannot convert between vector values of different size ('FourInts' (vector of 4 'int' values) and 'float')}}

  // When there is a vector and a scalar, conversions must be legal.
  (void)(four_ints ? four_floats : 3); // should work, ints can convert to floats.
  (void)(four_ints ? four_uints : e);  // expected-error {{cannot convert between vector values of different size ('FourUInts' (vector of 4 'unsigned int' values) and 'E')}}
  (void)(four_ints ? four_uints : se); // expected-error {{cannot convert between vector and non-scalar values ('FourUInts' (vector of 4 'unsigned int' values) and 'SE'}}

  (void)(two_ints ? two_ints : us);
  (void)(four_shorts ? four_shorts : uss);
  (void)(four_ints ? four_floats : us);
  (void)(four_ints ? four_floats : sint);
}

template <typename T1, typename T2>
struct is_same {
  static constexpr bool value = false;
};
template <typename T>
struct is_same<T, T> {
  static constexpr bool value = true;
};
template <typename T1, typename T2>
constexpr bool is_same_v = is_same<T1, T2>::value;
template <typename T>
T &&declval();

// Check the result types when given two vector types.
void ResultTypes() {
  // Vectors must be the same, but result is the type of the LHS/RHS.
  static_assert(is_same_v<TwoInts, decltype(declval<TwoInts>() ? declval<TwoInts>() : declval<TwoInts>())>);
  static_assert(is_same_v<TwoFloats, decltype(declval<TwoInts>() ? declval<TwoFloats>() : declval<TwoFloats>())>);

  // When both are scalars, converts to vectors of common type.
  static_assert(is_same_v<TwoUInts, decltype(declval<TwoInts>() ? declval<int>() : declval<unsigned int>())>);

  // Constant is allowed since it doesn't truncate, and should promote to float.
  static_assert(is_same_v<TwoFloats, decltype(declval<TwoInts>() ? declval<float>() : 5u)>);
  static_assert(is_same_v<TwoFloats, decltype(declval<TwoInts>() ? 5 : declval<float>())>);

  // when only 1 is a scalar, it should convert to a compatible type.
  static_assert(is_same_v<TwoFloats, decltype(declval<TwoInts>() ? declval<TwoFloats>() : declval<float>())>);
  static_assert(is_same_v<TwoInts, decltype(declval<TwoInts>() ? declval<TwoInts>() : declval<int>())>);
  static_assert(is_same_v<TwoFloats, decltype(declval<TwoInts>() ? declval<TwoFloats>() : 5)>);

  // For the Binary conditional operator, the result type is either the vector on the RHS (that fits the rules on size/count), or the scalar extended to the correct count.
  static_assert(is_same_v<TwoInts, decltype(declval<TwoInts>() ?: declval<TwoInts>())>);
  static_assert(is_same_v<TwoInts, decltype(declval<TwoInts>() ?: declval<int>())>);
}

template <typename Cond>
void dependent_cond(Cond C) {
  (void)(C ? 1 : 2);
}

template <typename Operand>
void dependent_operand(Operand C) {
  (void)(two_ints ? 1 : C);
  (void)(two_ints ? C : 1);
  (void)(two_ints ? C : C);
}

template <typename Cond, typename LHS, typename RHS>
void all_dependent(Cond C, LHS L, RHS R) {
  (void)(C ? L : R);
}

// Check dependent cases.
void Templates() {
  dependent_cond(two_ints);
  dependent_operand(two_floats);
  // expected-error@158 {{vector operands to the vector conditional must be the same type ('unsigned int __attribute__((ext_vector_type(4)))' (vector of 4 'unsigned int' values) and 'double __attribute__((ext_vector_type(4)))' (vector of 4 'double' values))}}}
  all_dependent(four_ints, four_uints, four_doubles); // expected-note {{in instantiation of}}

  // expected-error@158 {{vector operands to the vector conditional must be the same type ('unsigned int __attribute__((ext_vector_type(4)))' (vector of 4 'unsigned int' values) and 'unsigned int __attribute__((ext_vector_type(2)))' (vector of 2 'unsigned int' values))}}}
  all_dependent(four_ints, four_uints, two_uints); // expected-note {{in instantiation of}}
  all_dependent(four_ints, four_uints, four_uints);
}

using FourShortsVS = short __attribute__((__vector_size__(8)));

void mix_vector_types() {
  FourShortsVS vs;
  (vs == 1 ? four_shorts : four_shorts);
  // expected-error@-1 {{cannot mix vectors and extended vectors in a vector conditional}}

  (four_shorts == 1 ? vs : vs);
  // expected-error@-1 {{cannot mix vectors and extended vectors in a vector conditional}}

  (four_shorts == 1 ? four_shorts : vs);
  // expected-error@-1 {{vector operands to the vector conditional must be the same type ('FourShorts' (vector of 4 'short' values) and 'FourShortsVS' (vector of 4 'short' values))}}

  (four_shorts == 1 ? vs : four_shorts);
  // expected-error@-1 {{cannot mix vectors and extended vectors in a vector conditional}}

  (vs == 1 ? vs : four_shorts);
  // expected-error@-1 {{vector operands to the vector conditional must be the same type ('FourShortsVS' (vector of 4 'short' values) and 'FourShorts' (vector of 4 'short' values))}}

  (vs == 1 ? four_shorts : vs);
  // expected-error@-1 {{cannot mix vectors and extended vectors in a vector conditional}}
}
