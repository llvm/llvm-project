// RUN: %clang_cc1 -triple arm64-apple-ios -std=c++20 -fptrauth-calls -fptrauth-intrinsics -verify -fsyntax-only %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c++20 -fptrauth-calls -fptrauth-intrinsics -verify -fsyntax-only %s
// expected-no-diagnostics

#define AQ __ptrauth(1,1,50)
#define IQ __ptrauth(1,0,50)
#define AA [[clang::ptrauth_vtable_pointer(process_independent,address_discrimination,no_extra_discrimination)]]
#define IA [[clang::ptrauth_vtable_pointer(process_independent,no_address_discrimination,type_discrimination)]]
#define PA [[clang::ptrauth_vtable_pointer(process_dependent,no_address_discrimination,no_extra_discrimination)]]

template <class T>
struct Holder {
  T t_;
  bool operator==(const Holder&) const = default;
};

struct S1 {
  int * AQ p_;
  void *payload_;
  bool operator==(const S1&) const = default;
};
static_assert(__is_trivially_constructible(S1));
static_assert(!__is_trivially_constructible(S1, const S1&));
static_assert(!__is_trivially_assignable(S1, const S1&));
static_assert(__is_trivially_destructible(S1));
static_assert(!__is_trivially_copyable(S1));
static_assert(!__is_trivially_relocatable(S1));
static_assert(!__is_trivially_equality_comparable(S1));

static_assert(__is_trivially_constructible(Holder<S1>));
static_assert(!__is_trivially_constructible(Holder<S1>, const Holder<S1>&));
static_assert(!__is_trivially_assignable(Holder<S1>, const Holder<S1>&));
static_assert(__is_trivially_destructible(Holder<S1>));
static_assert(!__is_trivially_copyable(Holder<S1>));
static_assert(!__is_trivially_relocatable(Holder<S1>));
static_assert(!__is_trivially_equality_comparable(Holder<S1>));

struct S2 {
  int * IQ p_;
  void *payload_;
  bool operator==(const S2&) const = default;
};
static_assert(__is_trivially_constructible(S2));
static_assert(__is_trivially_constructible(S2, const S2&));
static_assert(__is_trivially_assignable(S2, const S2&));
static_assert(__is_trivially_destructible(S2));
static_assert(__is_trivially_copyable(S2));
static_assert(__is_trivially_relocatable(S2));
static_assert(__is_trivially_equality_comparable(S2));

static_assert(__is_trivially_constructible(Holder<S2>));
static_assert(__is_trivially_constructible(Holder<S2>, const Holder<S2>&));
static_assert(__is_trivially_assignable(Holder<S2>, const Holder<S2>&));
static_assert(__is_trivially_destructible(Holder<S2>));
static_assert(__is_trivially_copyable(Holder<S2>));
static_assert(__is_trivially_relocatable(Holder<S2>));
static_assert(__is_trivially_equality_comparable(Holder<S2>));

struct AA S3 {
  virtual void f();
  void *payload_;
  bool operator==(const S3&) const = default;
};

static_assert(!__is_trivially_constructible(S3));
static_assert(!__is_trivially_constructible(S3, const S3&));
static_assert(!__is_trivially_assignable(S3, const S3&));
static_assert(__is_trivially_destructible(S3));
static_assert(!__is_trivially_copyable(S3));
static_assert(!__is_trivially_relocatable(S3));
static_assert(!__is_trivially_equality_comparable(S3));

static_assert(!__is_trivially_constructible(Holder<S3>));
static_assert(!__is_trivially_constructible(Holder<S3>, const Holder<S3>&));
static_assert(!__is_trivially_assignable(Holder<S3>, const Holder<S3>&));
static_assert(__is_trivially_destructible(Holder<S3>));
static_assert(!__is_trivially_copyable(Holder<S3>));
static_assert(__is_trivially_relocatable(Holder<S3>));
static_assert(!__is_trivially_equality_comparable(Holder<S3>));

struct IA S4 {
  virtual void f();
  void *payload_;
  bool operator==(const S4&) const = default;
};

static_assert(!__is_trivially_constructible(S4));
static_assert(!__is_trivially_constructible(S4, const S4&));
static_assert(!__is_trivially_assignable(S4, const S4&));
static_assert(__is_trivially_destructible(S4));
static_assert(!__is_trivially_copyable(S4));
static_assert(!__is_trivially_relocatable(S4));
static_assert(!__is_trivially_equality_comparable(S4));

static_assert(!__is_trivially_constructible(Holder<S4>));
static_assert(!__is_trivially_constructible(Holder<S4>, const Holder<S4>&));
static_assert(!__is_trivially_assignable(Holder<S4>, const Holder<S4>&));
static_assert(__is_trivially_destructible(Holder<S4>));
static_assert(!__is_trivially_copyable(Holder<S4>));
static_assert(__is_trivially_relocatable(Holder<S4>));
static_assert(!__is_trivially_equality_comparable(Holder<S4>));

struct PA S5 {
  virtual void f();
  void *payload_;
  bool operator==(const S5&) const = default;
};

static_assert(!__is_trivially_constructible(S5));
static_assert(!__is_trivially_constructible(S5, const S5&));
static_assert(!__is_trivially_assignable(S5, const S5&));
static_assert(__is_trivially_destructible(S5));
static_assert(!__is_trivially_copyable(S5));
static_assert(!__is_trivially_relocatable(S5));
static_assert(!__is_trivially_equality_comparable(S5));

static_assert(!__is_trivially_constructible(Holder<S5>));
static_assert(!__is_trivially_constructible(Holder<S5>, const Holder<S5>&));
static_assert(!__is_trivially_assignable(Holder<S5>, const Holder<S5>&));
static_assert(__is_trivially_destructible(Holder<S5>));
static_assert(!__is_trivially_copyable(Holder<S5>));
static_assert(__is_trivially_relocatable(Holder<S5>));
static_assert(!__is_trivially_equality_comparable(Holder<S5>));
