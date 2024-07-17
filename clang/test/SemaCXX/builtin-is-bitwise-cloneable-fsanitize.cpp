// RUN: %clang_cc1 -triple x86_64-unknown-linux -DSANITIZER_ENABLED -fsanitize=address -fsanitize-address-field-padding=1 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux %s

struct S {
  ~S() {}
  virtual void foo() {}

  int buffer[1];
  int other_field = 0;
};

union U {
  S s;
};

struct Derived : S {};

static_assert(!__is_trivially_copyable(S));
#ifdef SANITIZER_ENABLED
// Don't allow memcpy when the struct has poisoned padding bits.
// The sanitizer adds posion padding bits to struct S.
static_assert(sizeof(S) > 16);
static_assert(!__is_bitwise_cloneable(S));
static_assert(sizeof(U) == sizeof(S)); // no padding bit for U.
static_assert(!__is_bitwise_cloneable(U));
static_assert(!__is_bitwise_cloneable(S[2]));
static_assert(!__is_bitwise_cloneable(Derived));
#else
static_assert(sizeof(S) == 16);
static_assert(__is_bitwise_cloneable(S));
static_assert(__is_bitwise_cloneable(U));
static_assert(__is_bitwise_cloneable(S[2]));
static_assert(__is_bitwise_cloneable(Derived));
#endif
