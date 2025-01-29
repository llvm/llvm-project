// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both -std=c++20 %s
// RUN: %clang_cc1 -verify=ref,both -std=c++20 %s

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
  return __builtin_constant_p(1) ? (intptr_t)p : (intptr_t)p; // expected-note {{cast that performs the conversions of a reinterpret_cast}}
}

/// This is from test/SemaCXX/builtin-constant-p.cpp, but it makes no sense.
/// ptr_to_int is called before bcp(), so it fails. GCC does not accept this either.
static_assert(bcp(ptr_to_int("foo"))); // expected-error {{not an integral constant expression}} \
                                       // expected-note {{in call to}}

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


