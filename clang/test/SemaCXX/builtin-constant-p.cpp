// RUN: %clang_cc1 -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify %s

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

struct Aggregate {
  int Value;
  constexpr Aggregate(int Value) : Value(Value) {}
};

constexpr Aggregate ConstantAggregate(42);
Aggregate NonConstantAggregate(42);
static_assert(bcp(ConstantAggregate));

void require_constant_aggregate(Aggregate Value) // expected-note {{candidate disabled: argument is not constant}}
    __attribute__((enable_if(__builtin_constant_p(Value),
                             "argument is not constant")));

void test_constant_aggregate() {
  require_constant_aggregate(ConstantAggregate);
  require_constant_aggregate(NonConstantAggregate); // expected-error {{no matching function for call to 'require_constant_aggregate'}}
}

constexpr int constant_aggregate_if() {
  constexpr Aggregate Value(42);
  if (__builtin_constant_p(Value))
    return Value.Value;
  return 0;
}
static_assert(constant_aggregate_if() == 42);

struct NonTrivial {
  int Value;
  constexpr NonTrivial(int Value) : Value(Value) {}
  constexpr NonTrivial(const NonTrivial &Other) : Value(Other.Value) {}
};

constexpr NonTrivial ConstantNonTrivial(42);
NonTrivial NonConstantNonTrivial(42);

constexpr int non_trivial_if(NonTrivial Value) {
  if (__builtin_constant_p(Value))
    return Value.Value;
  return 0;
}
static_assert(non_trivial_if(ConstantNonTrivial) == 42);

void require_constant_non_trivial(NonTrivial Value) // expected-note {{candidate disabled: argument is not constant}}
    __attribute__((enable_if(__builtin_constant_p(Value),
                             "argument is not constant")));

void test_constant_non_trivial() {
  require_constant_non_trivial(ConstantNonTrivial);
  require_constant_non_trivial(NonConstantNonTrivial); // expected-error {{no matching function for call to 'require_constant_non_trivial'}}
}

// Unions are record types, so a constant-evaluable union value is constant.
union ValueUnion {
  int Value;
  constexpr ValueUnion(int Value) : Value(Value) {}
  constexpr ValueUnion(const ValueUnion &Other) : Value(Other.Value) {}
};

constexpr ValueUnion ConstantUnion(42);
ValueUnion NonConstantUnion(42);
static_assert(bcp(ConstantUnion));
static_assert(!__builtin_constant_p(NonConstantUnion));

constexpr int constant_union_if() {
  constexpr ValueUnion Value(42);
  if (__builtin_constant_p(Value))
    return Value.Value;
  return 0;
}
static_assert(constant_union_if() == 42);

void require_constant_union(ValueUnion Value) // expected-note {{candidate disabled: argument is not constant}}
    __attribute__((enable_if(__builtin_constant_p(Value),
                             "argument is not constant")));

void test_constant_union() {
  require_constant_union(ConstantUnion);
  require_constant_union(NonConstantUnion); // expected-error {{no matching function for call to 'require_constant_union'}}
}

// `*this` is the class object. `this` passed into a callee is checked there.
struct ThisValue {
  int Value;
  constexpr ThisValue(int Value) : Value(Value) {}

  constexpr bool object_is_constant() const {
    return __builtin_constant_p(*this);
  }

  constexpr int value_if_constant() const {
    if (__builtin_constant_p(*this))
      return Value;
    return 0;
  }

  constexpr bool pass_this() const;
};

constexpr bool check_this_pointer(const ThisValue *Object) {
  return __builtin_constant_p(*Object);
}

constexpr bool ThisValue::pass_this() const {
  return check_this_pointer(this);
}

constexpr ThisValue ConstantThis(42);
ThisValue NonConstantThis(7);

static_assert(ConstantThis.object_is_constant());
static_assert(ConstantThis.value_if_constant() == 42);
static_assert(ConstantThis.pass_this());

constexpr bool check_this_object(const ThisValue &Object) {
  return Object.object_is_constant();
}
constexpr bool check_passed_this(const ThisValue &Object) {
  return Object.pass_this();
}
static_assert(!check_this_object(NonConstantThis));
static_assert(!check_passed_this(NonConstantThis));

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

void mutate_as_side_effect() {
  int a;
  static_assert(!__builtin_constant_p(((void)++a, 1)));
}

namespace dtor_side_effect {
  struct A {
    constexpr A() {}
    ~A();
  };
  static_assert(!__builtin_constant_p((A{}, 123)));
}

#if __cplusplus >= 202002L
namespace constexpr_dtor {
  struct A {
    int *p;
    constexpr ~A() { *p = 0; }
  };
  struct Q { int n; constexpr int *get() { return &n; } };
  static_assert(!__builtin_constant_p(((void)A{}, 123)));
  // FIXME: We should probably accept this. GCC does.
  // However, GCC appears to do so by running the destructors at the end of the
  // enclosing full-expression, which seems broken; running them at the end of
  // the evaluation of the __builtin_constant_p argument would be more
  // defensible.
  static_assert(!__builtin_constant_p(((void)A{Q().get()}, 123)));
}
#endif
