// RUN: %clang_cc1 -std=c++17 -verify %s

using intptr_t = __INTPTR_TYPE__;

// Test interaction of constexpr and __builtin_constant_p.

template<typename T> constexpr bool bcp(T t) {
  return __builtin_constant_p(t);
}
template<typename T> constexpr bool bcp_fold(T t) {
  return __builtin_constant_p(((void)(intptr_t)&t, t));
}

constexpr intptr_t ensure_fold_is_generally_not_enabled = // expected-error {{constant expression}}
    (intptr_t)&ensure_fold_is_generally_not_enabled; // expected-note {{cast}}

constexpr intptr_t ptr_to_int(const void *p) {
  return __builtin_constant_p(1) ? (intptr_t)p : (intptr_t)p;
}

constexpr int *int_to_ptr(intptr_t n) {
  return __builtin_constant_p(1) ? (int*)n : (int*)n;
}

int x;

// Integer and floating point constants encountered during constant expression
// evaluation are considered constant. So is nullptr_t.
static_assert(bcp(1));
static_assert(bcp_fold(1));
static_assert(bcp(1.0));
static_assert(bcp_fold(1.0));
static_assert(bcp(nullptr));
static_assert(bcp_fold(nullptr));

// Pointers to the start of strings are considered constant.
static_assert(bcp("foo"));
static_assert(bcp_fold("foo"));

// Null pointers are considered constant.
static_assert(bcp<int*>(nullptr));
static_assert(bcp_fold<int*>(nullptr));
static_assert(bcp<const char*>(nullptr));
static_assert(bcp_fold<const char*>(nullptr));

// Other pointers are not.
static_assert(!bcp(&x));
static_assert(!bcp_fold(&x));

// Pointers cast to integers follow the rules for pointers.
static_assert(bcp(ptr_to_int("foo")));
static_assert(bcp_fold(ptr_to_int("foo")));
static_assert(!bcp(ptr_to_int(&x)));
static_assert(!bcp_fold(ptr_to_int(&x)));

// Integers cast to pointers follow the integer rules.
static_assert(bcp(int_to_ptr(0)));
static_assert(bcp_fold(int_to_ptr(0)));
static_assert(bcp(int_to_ptr(123)));      // GCC rejects these due to not recognizing
static_assert(bcp_fold(int_to_ptr(123))); // the bcp conditional in 'int_to_ptr' ...
static_assert(__builtin_constant_p((int*)123)); // ... but GCC accepts this

// State mutations in the operand are not permitted.
//
// The rule GCC uses for this is not entirely understood, but seems to depend
// in some way on what local state is mentioned in the operand of
// __builtin_constant_p and where.
//
// We approximate GCC's rule by evaluating the operand in a speculative
// evaluation context; only state created within the evaluation can be
// modified.
constexpr int mutate1() {
  int n = 1;
  int m = __builtin_constant_p(++n);
  return n * 10 + m;
}
static_assert(mutate1() == 10);

// FIXME: GCC treats this as being non-constant because of the "n = 2", even
// though evaluation in the context of the enclosing constant expression
// succeeds without mutating any state.
constexpr int mutate2() {
  int n = 1;
  int m = __builtin_constant_p(n ? n + 1 : n = 2);
  return n * 10 + m;
}
static_assert(mutate2() == 11);

constexpr int internal_mutation(int unused) {
  int x = 1;
  ++x;
  return x;
}

constexpr int mutate3() {
  int n = 1;
  int m = __builtin_constant_p(internal_mutation(0));
  return n * 10 + m;
}
static_assert(mutate3() == 11);

constexpr int mutate4() {
  int n = 1;
  int m = __builtin_constant_p(n ? internal_mutation(0) : 0);
  return n * 10 + m;
}
static_assert(mutate4() == 11);

// FIXME: GCC treats this as being non-constant because of something to do with
// the 'n' in the argument to internal_mutation.
constexpr int mutate5() {
  int n = 1;
  int m = __builtin_constant_p(n ? internal_mutation(n) : 0);
  return n * 10 + m;
}
static_assert(mutate5() == 11);

constexpr int mutate_param(bool mutate, int &param) {
  mutate = mutate; // Mutation of internal state is OK
  if (mutate)
    ++param;
  return param;
}
constexpr int mutate6(bool mutate) {
  int n = 1;
  int m = __builtin_constant_p(mutate_param(mutate, n));
  return n * 10 + m;
}
// No mutation of state outside __builtin_constant_p: evaluates to true.
static_assert(mutate6(false) == 11);
// Mutation of state outside __builtin_constant_p: evaluates to false.
static_assert(mutate6(true) == 10);

// GCC strangely returns true for the address of a type_info object, despite it
// not being a pointer to the start of a string literal.
namespace std { struct type_info; }
static_assert(__builtin_constant_p(&typeid(int)));
