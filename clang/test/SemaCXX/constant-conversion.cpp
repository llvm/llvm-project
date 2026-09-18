// RUN: %clang_cc1 -fsyntax-only -Wbool-operation -verify -triple x86_64-apple-darwin %s
// RUN: %clang_cc1 -fsyntax-only -Wbool-operation -verify -triple x86_64-apple-darwin -fexperimental-new-constant-interpreter %s

// This file tests -Wconstant-conversion, a subcategory of -Wconversion
// which is on by default.

constexpr int nines() { return 99999; }

void too_big_for_char(int param) {
  char warn1 = false ? 0 : 99999;
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}
  char warn2 = false ? 0 : nines();
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}

  char warn3 = param > 0 ? 0 : 99999;
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}
  char warn4 = param > 0 ? 0 : nines();
  // expected-warning@-1 {{implicit conversion from 'int' to 'char' changes value from 99999 to -97}}

  char ok1 = true ? 0 : 99999;
  char ok2 = true ? 0 : nines();

  char ok3 = true ? 0 : 99999 + 1;
  char ok4 = true ? 0 : nines() + 1;
}

namespace GH223923 {

void static_const_conditions() {
  static const int zero = 0;
  static const int one = 1;
  signed char true_dead = zero ? 128 : 1;
  signed char false_dead = one ? 1 : 128;
  signed char true_live = one ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
  signed char false_live = zero ? 1 : 128;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
}

constexpr bool is_zero(int value) { return value == 0; }

void constexpr_conditions(int condition) {
  // Test constexpr conditions independently of constexpr destinations.
  constexpr bool never = false;
  constexpr bool always = true;
  signed char variable_true_dead = never ? 128 : 1;
  signed char variable_false_dead = always ? 1 : 128;
  signed char variable_true_live = always ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
  signed char variable_false_live = never ? 1 : 128;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}

  signed char function_true_dead = is_zero(1) ? 128 : 1;
  signed char function_false_dead = is_zero(2 - 2) ? 1 : 128;
  signed char function_true_live = is_zero(0) ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
  signed char function_false_live = is_zero(1 + 1) ? 1 : 128;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}

  // A constexpr function can also be called with a runtime argument.
  signed char function_true_maybe = is_zero(condition) ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
  signed char function_false_maybe = is_zero(condition) ? 1 : 128;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
}

void local_initializers(bool condition) {
  constexpr signed char constexpr_dead = false ? 128 : 1;
  constexpr signed char constexpr_live = true ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'const signed char' changes value from 128 to -128}}

  short short_dead = false ? 32768 : 1;
  int int_dead = false ? 2147483648LL : 1;

  short short_live = true ? 32768 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'short' changes value from 32768 to -32768}}
  int int_maybe = condition ? 2147483648LL : 1;
  // expected-warning@-1 {{implicit conversion from 'long long' to 'int' changes value from 2147483648 to -2147483648}}

  signed char nested_dead = true ? 1 : ((true ? 128 : 1) + 0);
  signed char nested_live = false ? 1 : ((true ? 128 : 1) + 0);
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
}

constexpr signed char global_dead = false ? 128 : 1;
constexpr signed char global_live = true ? 128 : 1;
// expected-warning@-1 {{implicit conversion from 'int' to 'const signed char' changes value from 128 to -128}}

struct MemberInitializers {
  signed char member_dead = false ? 128 : 1;
  signed char member_live = true ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}

  static const signed char static_dead = false ? 128 : 1;
  static const signed char static_live = true ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'const signed char' changes value from 128 to -128}}
};

void default_arg_dead(signed char = false ? 128 : 1);
void default_arg_live(signed char = true ? 128 : 1);
// expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}

// The larger truncation path must also respect the unselected operand, even
// in contexts where there is no function body to analyze for reachability.
void default_arg_truncation_dead(signed char = false ? 256 : 1);
void default_arg_truncation_live(signed char = true ? 256 : 1);
// expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 256 to 0}}

int consume(signed char);
signed char sink;
int total;

// Preserve the outer conditional's reachability when recursing through
// calls, comparisons, assignments, and compound assignments.
void call_dead(int = false ? consume(128) : 1);
void call_live(int = true ? consume(128) : 1);
// expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
void comparison_dead(bool = false ? (consume(128) == 0) : false);
void comparison_live(bool = true ? (consume(128) == 0) : false);
// expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
void assignment_dead(int = false ? (sink = 128) : 1);
void assignment_live(int = true ? (sink = 128) : 1);
// expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
void compound_assignment_dead(int = false ? (total += consume(128)) : 1);
void compound_assignment_live(int = true ? (total += consume(128)) : 1);
// expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}

struct NestedInitializers {
  bool comparison_dead = false ? (consume(128) == 0) : false;
  int assignment_dead = false ? (sink = 128) : 1;
  int compound_assignment_dead = false ? (total += consume(128)) : 1;
  signed char truncation_dead = false ? 256 : 1;
  signed char truncation_live = true ? 256 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 256 to 0}}
};

constexpr bool in_constant_evaluation() {
  return __builtin_is_constant_evaluated();
}

// Runtime reachability must not suppress conversions that take place during
// constant evaluation.
constexpr signed char constant_global_live = in_constant_evaluation() ? 128 : 1;
// expected-warning@-1 {{implicit conversion from 'int' to 'const signed char' changes value from 128 to -128}}
constexpr signed char constant_global_dead = in_constant_evaluation() ? 1 : 128;
static_assert(constant_global_live == -128, "");
static_assert(constant_global_dead == 1, "");

void constant_initializers() {
  constexpr signed char live = in_constant_evaluation() ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'const signed char' changes value from 128 to -128}}
  constexpr signed char dead = in_constant_evaluation() ? 1 : 128;
  static_assert(live == -128, "");
  static_assert(dead == 1, "");
}

// Neither operand can be ruled out when the same function can be evaluated
// both at runtime and at compile time.
constexpr signed char constant_function() {
  return in_constant_evaluation() ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
}
constexpr signed char runtime_function() {
  return in_constant_evaluation() ? 1 : 128;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
}
static_assert(constant_function() == -128, "");
static_assert(runtime_function() == 1, "");

constexpr int zero(signed char) { return 0; }

void binary_conditionals() {
  // The common expression of GNU's x ?: y is evaluated even when it is zero.
  int common_live = zero(128) ?: 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
  int common_dead = false ? (zero(128) ?: 1) : 1;

  signed char true_live = 128 ?: 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
  signed char false_live = 0 ?: 128;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
  signed char false_dead = 1 ?: 128;
}

template <bool Condition> void template_conditional() {
  signed char value = Condition ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
}

template <bool Condition> void template_false_operand() {
  signed char value = Condition ? 1 : 128;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
}

template <typename T> void sizeof_condition() {
  signed char true_operand = sizeof(T) == 1 ? 128 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
  signed char false_operand = sizeof(T) == 1 ? 1 : 128;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
}

void instantiate_conditionals() {
  template_conditional<false>();
  template_conditional<true>();
  // expected-note@-1 {{in instantiation of function template specialization 'GH223923::template_conditional<true>' requested here}}
  template_false_operand<true>();
  template_false_operand<false>();
  // expected-note@-1 {{in instantiation of function template specialization 'GH223923::template_false_operand<false>' requested here}}
  sizeof_condition<char>();
  // expected-note@-1 {{in instantiation of function template specialization 'GH223923::sizeof_condition<char>' requested here}}
  sizeof_condition<int>();
  // expected-note@-1 {{in instantiation of function template specialization 'GH223923::sizeof_condition<int>' requested here}}
}

// Suppressing constant-conversion diagnostics must not skip other checks in
// unselected operands.
int unrelated_diagnostic(bool b) {
  return false ? ~b : 0;
  // expected-warning@-1 {{bitwise negation of a boolean expression}}
}

} // namespace GH223923

void test_bitfield() {
  struct S {
    int one_bit : 1;
  } s;

  s.one_bit = 1;    // expected-warning {{implicit truncation from 'int' to a one-bit wide bit-field changes value from 1 to -1}}
  s.one_bit = true; // no-warning
}
