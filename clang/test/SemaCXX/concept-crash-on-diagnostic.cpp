// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

template <typename Iterator> class normal_iterator {};

template <typename From, typename To> struct is_convertible {};

template <typename From, typename To>
inline constexpr bool is_convertible_v = is_convertible<From, To>::value; // expected-error {{no member named 'value' in 'is_convertible<bool, bool>'}}

template <typename From, typename To>
concept convertible_to = is_convertible_v<From, To>; // #1

template <typename IteratorL, typename IteratorR>
  requires requires(IteratorL lhs, IteratorR rhs) { // #2
    { lhs == rhs } -> convertible_to<bool>; // #3
  }
constexpr bool compare(normal_iterator<IteratorL> lhs, normal_iterator<IteratorR> rhs) { // #4
  return false;
}

class Object;

void function() {
  normal_iterator<Object *> begin, end;
  compare(begin, end); // expected-error {{no matching function for call to 'compare'}} #5
}

// expected-note@#1 {{in instantiation of variable template specialization 'is_convertible_v<bool, bool>' requested here}}
// expected-note@#1 {{substituting template arguments into constraint expression here}}
// expected-note@#3 {{checking the satisfaction of concept 'convertible_to<bool, bool>'}}
// expected-note@#2 {{substituting template arguments into constraint expression here}}
// expected-note@#5 {{checking constraint satisfaction for template 'compare<Object *, Object *>'}}
// expected-note@#5 {{in instantiation of function template specialization 'compare<Object *, Object *>' requested here}}

// expected-note@#4 {{candidate template ignored: constraints not satisfied [with IteratorL = Object *, IteratorR = Object *]}}
// We don't know exactly the substituted type for `lhs == rhs`, thus a placeholder 'expr-type' is emitted.
// expected-note@#3 {{because 'convertible_to<expr-type, bool>' would be invalid}}
