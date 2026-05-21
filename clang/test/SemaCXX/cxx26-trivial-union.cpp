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
// Dtor is deleted: user-provided default ctor is non-trivial
// (see C++26 [class.dtor]p7).
union U5 { // cxx26-note {{not trivial}}
  U5() : nt() {}
  NonTrivialDtor nt; // precxx26-note {{non-trivial}}
};

void test_u5_destroy(U5 *p) { p->~U5(); } // cxx26-error {{deleted}} precxx26-error {{deleted}}

// ===== Test 6: Feature test macro =====
#if __cplusplus > 202302L
static_assert(__cpp_trivial_union >= 202603L);
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

// ===== Test 11: No default ctor (suppressed by user-declared ctor) =====
// When the implicit default ctor is suppressed (not deleted), the
// [class.dtor]p7 bullet 1 rule does not apply. The union can't be
// default-initialized, so no risk of indeterminate active member.
// Destructor is trivial (see C++26 [class.dtor]p7).
#if __cplusplus > 202302L
union U11 {
  U11(int);
  NonTrivialDtor nt;
};
static_assert(__is_trivially_destructible(U11));
U11 u11(1);

// ===== Test 12: Deleted default ctor (dtor deleted per [class.dtor]p7) =====
union U12 { // cxx26-note {{deleted function}}
  U12() = delete;
  U12(int);
  NonTrivialDtor nt;
};
U12 u12(1); // cxx26-error {{deleted}}

// ===== Test 13: Defaulted ctor => trivial, dtor NOT deleted =====
union U13 {
  U13() = default;
  NonTrivialDtor nt;
  U13 *next = nullptr;
};
U13 u13;

// ===== Test 14: Array member with DMI + non-trivial dtor ([class.dtor]p7 bullet 2) =====
struct NonTrivialInt {
  int i;
  constexpr NonTrivialInt(int i) : i(i) {}
  constexpr ~NonTrivialInt() {}
};

union U14 {
  NonTrivialInt arr[2] = {1, 2}; // cxx26-note {{non-trivial}}
};
U14 u14; // cxx26-error {{deleted}}

// ===== Test 15: Anonymous struct in union, no DMI => NOT deleted =====
union U15 {
  struct {
    NonTrivialDtor x;
  };
};
U15 u15;

// ===== Test 16: Anonymous struct in union, with DMI => deleted =====
union U16 {
  struct {
    NonTrivialInt x = 1; // cxx26-note {{non-trivial}}
  };
};
U16 u16; // cxx26-error {{deleted}}

// ===== Test 17: struct containing anonymous union =====
struct S17 {
  union {
    NonTrivialDtor x;
  };
};
S17 s17;

// ===== Test 18: Deeply nested union-struct-union-struct-union =====
union U18 {
  struct {
    union {
      struct {
        union {
          NonTrivialDtor x;
        };
      };
    };
  };
};
U18 u18;

// ===== Test 19: struct-union-struct-union-struct nesting =====
struct S19 {
  union {
    struct {
      union {
        struct {
          NonTrivialDtor x;
        };
      };
    };
  };
};
S19 s19;

// ===== Test 20: Anonymous union inside union =====
union U20 {
  union {
    NonTrivialDtor x;
  };
};
U20 u20;

// ===== Test 21: Deleted destructor member (p7.2, not union-specific) =====
// p7.2 still applies: deleted/inaccessible member dtor => union dtor deleted.
struct DeletedDtor {
  ~DeletedDtor() = delete; // cxx26-note {{deleted here}} precxx26-note {{deleted here}}
};

union U21 {
  DeletedDtor a; // cxx26-note {{deleted destructor}} precxx26-note {{deleted destructor}}
};
void test_u21(U21 *p) { p->~U21(); } // cxx26-error {{deleted}} precxx26-error {{deleted}}

// ===== Test 22: Constexpr evaluation of trivial union =====
constexpr int constexpr_test() {
  U2 u;
  u.k = 42;
  return u.k;
}
static_assert(constexpr_test() == 42);

// ===== Test 23: Ambiguous default ctor => dtor deleted =====
// Multiple constructors with default arguments make default-initialization
// ambiguous, so dtor is deleted per [class.dtor]p7 bullet 1.
union U23 { // cxx26-note {{ambiguous}}
  U23(int = 0);
  U23(double = 0.0);
  NonTrivialDtor nt;
};
U23 u23(42); // cxx26-error {{deleted}}
#endif
