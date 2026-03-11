// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s

// P3726R1: __builtin_start_lifetime and constituent values tests

namespace std {
  using size_t = decltype(sizeof(0));
}
void* operator new(std::size_t, void* p) noexcept { return p; }

// ===== Type checking tests =====

struct Agg { int x; int y; };
struct NonAgg {
  NonAgg(int);
  int x;
};
struct AggWithUserDtor {
  int x;
  ~AggWithUserDtor();
};

// type checking for __builtin_start_lifetime is done via consteval contexts.
consteval void check_agg() {
  Agg a;
  __builtin_start_lifetime(&a); // OK
}
consteval void check_array() {
  int arr[4];
  __builtin_start_lifetime(&arr); // OK
}

consteval void check_scalar() {
  int x = 0;
  __builtin_start_lifetime(&x); // expected-error {{pointer to a complete implicit-lifetime aggregate type}}
}

consteval void check_non_agg() {
  // NonAgg is not constructible without an argument, can't be a local here.
  // Just test the pointer type check with an invalid construct.
}

consteval void check_user_dtor() {
  AggWithUserDtor awd;
  __builtin_start_lifetime(&awd); // expected-error {{pointer to a complete implicit-lifetime aggregate type}}
}

// ===== Constexpr evaluation tests =====

// Test: start_lifetime on array member of union, then placement new elements
consteval int test_start_lifetime_array() {
  struct S {
    union { int storage[4]; };
    int size = 0;
  };
  S s;
  __builtin_start_lifetime(&s.storage);
  // Now storage is the active member, but no elements are within lifetime.
  ::new (&s.storage[0]) int(10);
  ::new (&s.storage[1]) int(20);
  s.size = 2;
  return s.storage[0] + s.storage[1]; // 30
}
static_assert(test_start_lifetime_array() == 30);

// Test: start_lifetime is no-op if already within lifetime
consteval int test_start_lifetime_noop() {
  struct S {
    union { int storage[2]; };
  };
  S s;
  __builtin_start_lifetime(&s.storage);
  ::new (&s.storage[0]) int(42);
  // Call again - should be a no-op since storage is already active
  __builtin_start_lifetime(&s.storage);
  return s.storage[0]; // Still 42
}
static_assert(test_start_lifetime_noop() == 42);

// Test: start_lifetime on struct member of union
consteval int test_start_lifetime_struct() {
  struct Inner { int a; int b; };
  union U { Inner inner; int x; };
  U u;
  __builtin_start_lifetime(&u.inner);
  // inner is now active but its members aren't initialized yet
  ::new (&u.inner.a) int(1);
  ::new (&u.inner.b) int(2);
  return u.inner.a + u.inner.b;
}
static_assert(test_start_lifetime_struct() == 3);

// ===== Constituent values: array with holes in union =====
// P3726R1 [expr.const]p2: array elements not within their lifetime
// in a union are inactive union subobjects and should be skipped.

struct CVResult {
  union { int arr[4]; };
  int size;
};

consteval CVResult test_constituent_values() {
  CVResult s;
  s.size = 2;
  __builtin_start_lifetime(&s.arr);
  ::new (&s.arr[0]) int(100);
  ::new (&s.arr[1]) int(200);
  // arr[2] and arr[3] are not within their lifetime — that's OK per P3726R1.
  return s;
}
// This should be a valid constexpr variable even though arr[2] and arr[3]
// are not initialized — they are inactive union subobjects per P3726R1.
constexpr auto cv_result = test_constituent_values();
static_assert(cv_result.arr[0] == 100);
static_assert(cv_result.arr[1] == 200);
static_assert(cv_result.size == 2);
