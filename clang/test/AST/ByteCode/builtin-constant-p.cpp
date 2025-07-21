// RUN: %clang_cc1 -std=c++20 -verify=expected,both %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++20 -verify=ref,both      %s

using intptr_t = __INTPTR_TYPE__;

static_assert(__builtin_constant_p(12), "");
static_assert(__builtin_constant_p(1.0), "");

constexpr int I = 100;
static_assert(__builtin_constant_p(I), "");
static_assert(__builtin_constant_p(I + 10), "");
static_assert(__builtin_constant_p(I + 10.0), "");
static_assert(__builtin_constant_p(nullptr), "");
static_assert(__builtin_constant_p(&I), ""); // both-error {{failed due to requirement}}
static_assert(__builtin_constant_p((void)I), ""); // both-error {{failed due to requirement}}

extern int z;
constexpr int foo(int &a) {
  return __builtin_constant_p(a);
}
static_assert(!foo(z));

static_assert(__builtin_constant_p(__builtin_constant_p(1)));

constexpr bool nested(int& a) {
  return __builtin_constant_p(__builtin_constant_p(a));
}
static_assert(nested(z));

constexpr bool Local() {
  int z = 10;
  return __builtin_constant_p(z);
}
static_assert(Local());

constexpr bool Local2() {
  int z = 10;
  return __builtin_constant_p(&z);
}
static_assert(!Local2());

constexpr bool Parameter(int a) {
  return __builtin_constant_p(a);
}
static_assert(Parameter(10));

constexpr bool InvalidLocal() {
  int *z;
  {
    int b = 10;
    z = &b;
  }
  return __builtin_constant_p(z);
}
static_assert(!InvalidLocal());

template<typename T> constexpr bool bcp(T t) {
  return __builtin_constant_p(t);
}

constexpr intptr_t ptr_to_int(const void *p) {
  return __builtin_constant_p(1) ? (intptr_t)p : (intptr_t)p;
}

static_assert(bcp(ptr_to_int("foo")));

constexpr bool AndFold(const int &a, const int &b) {
  return __builtin_constant_p(a && b);
}

static_assert(AndFold(10, 20));
static_assert(!AndFold(z, 10));
static_assert(!AndFold(10, z));


struct F {
  int a;
};

constexpr F f{12};
static_assert(__builtin_constant_p(f.a));

constexpr bool Member() {
  F f;
  return __builtin_constant_p(f.a);
}
static_assert(!Member());

constexpr bool Discard() {
  (void)__builtin_constant_p(10);
  return true;
}
static_assert(Discard());

static_assert(__builtin_constant_p((int*)123));

constexpr void func() {}
static_assert(!__builtin_constant_p(func));

/// This is from SemaCXX/builtin-constant-p and GCC agrees with the bytecode interpreter.
constexpr int mutate1() {
  int n = 1;
  int m = __builtin_constant_p(++n);
  return n * 10 + m;
}
static_assert(mutate1() == 21); // ref-error {{static assertion failed}} \
                                // ref-note {{evaluates to '10 == 21'}}

/// Similar for this. GCC agrees with the bytecode interpreter.
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
static_assert(mutate6(false) == 11);
static_assert(mutate6(true) == 21); // ref-error {{static assertion failed}} \
                                    // ref-note {{evaluates to '10 == 21'}}

#define fold(x) (__builtin_constant_p(x) ? (x) : (x))
void g() {
  /// f will be revisited when evaluating the static_assert, since it's
  /// a local variable. But it should be visited in a non-constant context.
  const float f = __builtin_is_constant_evaluated();
  static_assert(fold(f == 0.0f));
}

void test17(void) {
#define ASSERT(...) { enum { folded = (__VA_ARGS__) }; int arr[folded ? 1 : -1]; }
#define T(...) ASSERT(__builtin_constant_p(__VA_ARGS__))
#define F(...) ASSERT(!__builtin_constant_p(__VA_ARGS__))

  T(3i + 5);
  T("string literal");
  F("string literal" + 1); // both-warning {{adding}} \
                           // both-note {{use array indexing}}
}

/// FIXME
static void foo(int i) __attribute__((__diagnose_if__(!__builtin_constant_p(i), "not constant", "error"))) // expected-note {{from}}
{
}
static void bar(int i) {
  foo(15); // expected-error {{not constant}}
}
