// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s
// FIXME: Enable for bytecode interpreter once union array member activation
// is supported: %clang_cc1 -std=c++26 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

// P3726R2: __builtin_start_lifetime and constituent values tests

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

// ===== Error checking tests =====

consteval bool check_null_pointer() {
  Agg *p = nullptr;
  __builtin_start_lifetime(p); // expected-note {{cannot be called with a null pointer}}
  return true;
}
static_assert(check_null_pointer()); // expected-error {{constant expression}} \
                                     // expected-note {{in call to}}

consteval bool check_one_past_the_end() {
  Agg a[2];
  __builtin_start_lifetime(&a[2]); // expected-note {{cannot be called with a one-past-the-end pointer}}
  return true;
}
static_assert(check_one_past_the_end()); // expected-error {{constant expression}} \
                                         // expected-note {{in call to}}

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
// C++26 [expr.const]p2: A union elemental subobject is a direct member of a
// union or an element of an array that is a union elemental subobject.
// An inactive union elemental subobject is one not within its lifetime.

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
  // arr[2] and arr[3] are not within their lifetime — that's OK per P3726R2.
  return s;
}
// This should be a valid constexpr variable even though arr[2] and arr[3]
// are not initialized — they are inactive union elemental subobjects.
constexpr auto cv_result = test_constituent_values();
static_assert(cv_result.arr[0] == 100);
static_assert(cv_result.arr[1] == 200);
static_assert(cv_result.size == 2);

// CWG guidance (Croyden): starting the lifetime of a multi-dimensional array
// recursively starts the lifetime of each nested sub-array, but not scalar
// elements. The wording will be fixed to make this clear.
consteval int test_multidim_single_start() {
  struct S {
    union {
      int storage[2][3];
    };
  };
  S s;
  __builtin_start_lifetime(&s.storage);
  // No need to start_lifetime each sub-array — done recursively.
  ::new (&s.storage[0][0]) int(1);
  ::new (&s.storage[0][1]) int(2);
  ::new (&s.storage[1][0]) int(10);
  return s.storage[0][0] + s.storage[0][1] + s.storage[1][0];
}
static_assert(test_multidim_single_start() == 13);

// 3-dimensional array: start_lifetime recurses through all sub-array levels.
consteval int test_3d_array() {
  struct S {
    union {
      int storage[2][2][2];
    };
  };
  S s;
  __builtin_start_lifetime(&s.storage);
  ::new (&s.storage[0][0][0]) int(1);
  ::new (&s.storage[1][1][1]) int(7);
  return s.storage[0][0][0] + s.storage[1][1][1];
}
static_assert(test_3d_array() == 8);
