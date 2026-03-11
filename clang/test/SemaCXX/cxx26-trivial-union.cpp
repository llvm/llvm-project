// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify=cxx26 %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify=precxx26 %s

// P3074R7: trivial unions

struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial&);
  NonTrivial& operator=(const NonTrivial&);
  ~NonTrivial();
};

struct NonTrivialDtor {
  ~NonTrivialDtor();
};

// ===== Test 1: Basic union with non-trivial member =====
// P3074: default ctor and dtor should be trivial, not deleted.
union U1 {
  NonTrivial nt; // precxx26-note 2{{non-trivial}}
};

#if __cplusplus > 202302L
static_assert(__is_trivially_constructible(U1));
static_assert(__is_trivially_destructible(U1));
U1 test_u1;
#else
U1 test_u1_pre; // precxx26-error {{deleted}}
void destroy_u1(U1 *p) { p->~U1(); } // precxx26-error {{deleted}}
#endif

// ===== Test 2: Union with non-trivial member and int =====
union U2 {
  NonTrivial nt;
  int k;
};

#if __cplusplus > 202302L
static_assert(__is_trivially_constructible(U2));
static_assert(__is_trivially_destructible(U2));
#endif

// ===== Test 3: Union with DMI on member with non-trivial dtor =====
// P3074: dtor is deleted because DMI + non-trivial dtor on same member.
union U3_deleted_dtor {
  NonTrivialDtor ntd = {}; // cxx26-note {{non-trivial}} precxx26-note {{non-trivial}}
};

void test_u3_destroy(U3_deleted_dtor *p) {
  p->~U3_deleted_dtor(); // cxx26-error {{deleted}} precxx26-error {{deleted}}
}

// ===== Test 4: Union with DMI on non-class member =====
// DMI on int, but NonTrivial has no DMI => dtor should NOT be deleted.
union U4 {
  NonTrivial nt; // precxx26-note {{non-trivial}}
  int k = 42;
};

#if __cplusplus > 202302L
// Despite non-trivial default ctor (due to DMI), destructor is NOT deleted
// because the member with DMI (k) is int (trivially destructible).
static_assert(__is_trivially_destructible(U4));
#else
void destroy_u4(U4 *p) { p->~U4(); } // precxx26-error {{deleted}}
#endif

// ===== Test 5: Union with user-provided default constructor =====
union U5 { // cxx26-note {{user-provided}}
  U5() : nt() {}
  NonTrivialDtor nt;
};

#if __cplusplus > 202302L
// P3074 (7.x.1): user-provided default ctor => destructor is deleted.
void test_u5_destroy(U5 *p) { p->~U5(); } // cxx26-error {{deleted}}
#endif

// ===== Test 6: Feature test macro =====
#if __cplusplus > 202302L
static_assert(__cpp_trivial_union >= 202502L);
#else
#ifdef __cpp_trivial_union
#error "should not have __cpp_trivial_union in C++23"
#endif
#endif

// ===== Test 7: Trivial union (no change from status quo) =====
union U7 {
  int a;
  float b;
};

static_assert(__is_trivially_constructible(U7));
static_assert(__is_trivially_destructible(U7));

// ===== Test 8: Array member in union =====
union U8 {
  NonTrivial arr[4];
};

#if __cplusplus > 202302L
static_assert(__is_trivially_constructible(U8));
static_assert(__is_trivially_destructible(U8));
#endif

// ===== Test 9: Paper example - string with DMI =====
struct FakeString {
  FakeString(const char*);
  FakeString(const FakeString&);
  FakeString& operator=(const FakeString&);
  ~FakeString();
};

union PaperU2 {
  FakeString s = "hello"; // cxx26-note {{non-trivial}} precxx26-note {{non-trivial}}
};

void test_paper_u2(PaperU2 *p) {
  p->~PaperU2(); // cxx26-error {{deleted}} precxx26-error {{deleted}}
}

// ===== Test 10: Paper example U4 - DMI on pointer, non-trivial string =====
union PaperU4 {
  FakeString s; // precxx26-note {{non-trivial}}
  PaperU4 *next = nullptr;
};

#if __cplusplus > 202302L
static_assert(__is_trivially_destructible(PaperU4));
#else
void destroy_paper_u4(PaperU4 *p) { p->~PaperU4(); } // precxx26-error {{deleted}}
#endif
