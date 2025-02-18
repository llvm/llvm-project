// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s


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
