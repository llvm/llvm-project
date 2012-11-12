// RUN: %clang_cc1 %s -std=c++11 -fsyntax-only -verify -triple x86_64-linux-gnu

struct S;
constexpr int extract(const S &s);

struct S {
  constexpr S() : n(extract(*this)), m(0) {} // expected-note {{in call to 'extract(s1)'}}
  constexpr S(int k) : n(k), m(extract(*this)) {}
  int n, m;
};

constexpr int extract(const S &s) { return s.n; } // expected-note {{read of uninitialized object is not allowed in a constant expression}}

constexpr S s1; // ok
void f() {
  constexpr S s1; // expected-error {{constant expression}} expected-note {{in call to 'S()'}}
  constexpr S s2(10);
}

typedef __attribute__((vector_size(16))) int vector_int;

struct T {
  constexpr T() : arr() {}
  int arr[4];
};
struct U : T {
  constexpr U(const int *p) : T(), another(), p(p) {}
  constexpr U(const U &u) : T(), another(), p(u.p) {}
  T another;
  const int *p;
};
constexpr U u1(&u1.arr[2]);

constexpr int test_printing(int a, float b, _Complex int c, _Complex float d,
                            int *e, int &f, vector_int g, U h) {
  return *e; // expected-note {{read of non-constexpr variable 'u2'}}
}
U u2(0); // expected-note {{here}}
static_assert(test_printing(12, 39.762, 3 + 4i, 12.9 + 3.6i, &u2.arr[4], u2.another.arr[2], (vector_int){5, 1, 2, 3}, u1) == 0, ""); // \
expected-error {{constant expression}} \
expected-note {{in call to 'test_printing(12, 3.976200e+01, 3+4i, 1.290000e+01+3.600000e+00i, &u2.T::arr[4], u2.another.arr[2], {5, 1, 2, 3}, {{{}}, {{}}, &u1.T::arr[2]})'}}

struct V {
  // FIXME: when we can generate these as constexpr constructors, remove the
  // explicit definitions.
  constexpr V() : arr{[255] = 42} {}
  constexpr V(const V &v) : arr{[255] = 42} {}
  int arr[256];
};
constexpr V v;
constexpr int get(const int *p) { return *p; } // expected-note {{read of dereferenced one-past-the-end pointer}}
constexpr int passLargeArray(V v) { return get(v.arr+256); } // expected-note {{in call to 'get(&v.arr[256])'}}
static_assert(passLargeArray(v) == 0, ""); // expected-error {{constant expression}} expected-note {{in call to 'passLargeArray({{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...}})'}}

union Union {
  constexpr Union(int n) : b(n) {}
  constexpr Union(const Union &u) : b(u.b) {}
  int a, b;
};
constexpr Union myUnion = 76;

constexpr int badness(Union u) { return u.a + u.b; } // expected-note {{read of member 'a' of union with active member 'b'}}
static_assert(badness(myUnion), ""); // expected-error {{constant expression}} \
        expected-note {{in call to 'badness({.b = 76})'}}

struct MemPtrTest {
  int n;
  void f();
};
MemPtrTest mpt; // expected-note {{here}}
constexpr int MemPtr(int (MemPtrTest::*a), void (MemPtrTest::*b)(), int &c) {
  return c; // expected-note {{read of non-constexpr variable 'mpt'}}
}
static_assert(MemPtr(&MemPtrTest::n, &MemPtrTest::f, mpt.*&MemPtrTest::n), ""); // expected-error {{constant expression}} \
expected-note {{in call to 'MemPtr(&MemPtrTest::n, &MemPtrTest::f, mpt.n)'}}

template<typename CharT>
constexpr CharT get(const CharT *p) { return p[-1]; } // expected-note 5{{}}

constexpr char c = get("test\0\\\"\t\a\b\234"); // \
  expected-error {{}} expected-note {{"test\000\\\"\t\a\b\234"}}
constexpr char c8 = get(u8"test\0\\\"\t\a\b\234"); // \
  expected-error {{}} expected-note {{u8"test\000\\\"\t\a\b\234"}}
constexpr char16_t c16 = get(u"test\0\\\"\t\a\b\234\u1234"); // \
  expected-error {{}} expected-note {{u"test\000\\\"\t\a\b\234\u1234"}}
constexpr char32_t c32 = get(U"test\0\\\"\t\a\b\234\u1234\U0010ffff"); // \
  expected-error {{}} expected-note {{U"test\000\\\"\t\a\b\234\u1234\U0010FFFF"}}
constexpr wchar_t wc = get(L"test\0\\\"\t\a\b\234\u1234\xffffffff"); // \
  expected-error {{}} expected-note {{L"test\000\\\"\t\a\b\234\x1234\xFFFFFFFF"}}

constexpr char32_t c32_err = get(U"\U00110000"); // expected-error {{invalid universal character}}

typedef decltype(sizeof(int)) LabelDiffTy;
constexpr LabelDiffTy mulBy3(LabelDiffTy x) { return x * 3; } // expected-note {{subexpression}}
void LabelDiffTest() {
  static_assert(mulBy3((LabelDiffTy)&&a-(LabelDiffTy)&&b) == 3, ""); // expected-error {{constant expression}} expected-note {{call to 'mulBy3(&&a - &&b)'}}
  a:b:return;
}

constexpr bool test_bool_printing(bool b) { return 1 / !(2*b | !(2*b)); } // expected-note 2{{division by zero}}
constexpr bool test_bool_0 = test_bool_printing(false); // expected-error {{constant expr}} expected-note {{in call to 'test_bool_printing(false)'}}
constexpr bool test_bool_1 = test_bool_printing(true); // expected-error {{constant expr}} expected-note {{in call to 'test_bool_printing(true)'}}
