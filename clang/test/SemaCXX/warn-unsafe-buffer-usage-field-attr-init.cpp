// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions -verify %s
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -verify -verify-ignore-unexpected=note %s

// Initializing a field annotated with [[clang::unsafe_buffer_usage]] writes to
// it, so it is diagnosed like an assignment to the field.

struct J {
  J();
  J(int);
  void mutate();
};

struct S {
  int x;
  [[clang::unsafe_buffer_usage]] int *ptr;
  [[clang::unsafe_buffer_usage]] J j;
};

void test_member_access(S s, int *p) {
  s.ptr = p; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  s.j.mutate(); // expected-warning{{field 'j' prone to unsafe buffer manipulation}}
}

class C {
public:
  C() : ptr(nullptr) {} // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  C(int *p) : ptr(p) {} // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  C(int v) : j(v) {} // expected-warning{{field 'j' prone to unsafe buffer manipulation}}
  C(char) : j() {} // expected-warning{{field 'j' prone to unsafe buffer manipulation}}
  C(int *p, int v)
      : ptr(p), // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
        j(v) {} // expected-warning{{field 'j' prone to unsafe buffer manipulation}}

  // Members that are not initialized explicitly are not diagnosed.
  C(long) : x(0) {}
  C(short) {}

private:
  int x;
  [[clang::unsafe_buffer_usage]] int *ptr;
  [[clang::unsafe_buffer_usage]] J j;
  [[clang::unsafe_buffer_usage]] int sz = 0;
};

void test_constructors(int *p) {
  C c1;
  C c2(p);
  C c3 = c2;
}

struct WithAnon {
  WithAnon(int *p) : q(p) {} // expected-warning{{field 'q' prone to unsafe buffer manipulation}}
  union {
    int *u;
    [[clang::unsafe_buffer_usage]] int *q;
  };
};

void test_aggregate(int *p) {
  S a = {0, p}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  S b = {.x = 1};
  S c = {.ptr = p}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  S d = {.j = J(3)}; // expected-warning{{field 'j' prone to unsafe buffer manipulation}}
  S e = {.j = {}}; // expected-warning{{field 'j' prone to unsafe buffer manipulation}}
  S o = {.ptr = {}}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  // expected-warning@+1 2{{prone to unsafe buffer manipulation}}
  S f = {0, p, J(3)};
  S g = {};
  S h{};
  S i(0, p); // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  S k(0);
  S l = a;
  S m = S{.ptr = p}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  S *n = new S{0, p}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
}

struct T {
  S s;
  int y;
};

void test_nested(int *p) {
  T a = {{0, p}, 1}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  T b = {.s = {.ptr = p}}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  T c = {.s.ptr = p}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}} expected-warning{{nested designators are a C99 extension}}
  T d = {.s = {.x = 1}, .y = 2};
  T e = {.y = 2};
  // Brace elision.
  T f = {0, p}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  T g = {0};
  S arr[2] = {{0, p}, {.x = 1}}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  S arr2[2] = {0, p}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
}

struct Base {
  [[clang::unsafe_buffer_usage]] int *bp;
};

struct Derived : Base {
  int n;
};

void test_base(int *p) {
  Derived a = {{p}, 1}; // expected-warning{{field 'bp' prone to unsafe buffer manipulation}}
  Derived b = {p, 1}; // expected-warning{{field 'bp' prone to unsafe buffer manipulation}}
  Derived c = {{}, 1};
}

union U {
  int a;
  [[clang::unsafe_buffer_usage]] int *b;
};

struct A {
  union {
    int *u;
    [[clang::unsafe_buffer_usage]] int *q;
  };
};

void test_union(int *p) {
  U u1 = {1};
  U u2 = {.b = p}; // expected-warning{{field 'b' prone to unsafe buffer manipulation}}
  A a1 = {.q = p}; // expected-warning{{field 'q' prone to unsafe buffer manipulation}}
  A a2 = {.u = p};
}

struct Fwd;
struct Fwd {
  int : 4;
  int bits : 4;
  [[clang::unsafe_buffer_usage]] int *ptr;
};

void test_bitfields(int *p) {
  Fwd a = {1, p}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  Fwd b = {.ptr = p}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
}

// The attribute on an anonymous struct does not apply to its members.
struct AnonAttr {
  [[clang::unsafe_buffer_usage]] struct {
    int *a;
  };
};

void test_anon_attr(int *p) {
  AnonAttr a = {.a = p};
}

template <typename T> struct TS {
  [[clang::unsafe_buffer_usage]] T *ptr;
  unsigned n;
};

template <typename T> TS<T> make(T *p, unsigned n) {
  return {p, n}; // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
}

template <typename T> struct TC {
  TC(T *p) : ptr(p) {} // expected-warning{{field 'ptr' prone to unsafe buffer manipulation}}
  [[clang::unsafe_buffer_usage]] T *ptr;
};

void test_templates(int *p) {
  make(p, 1);
  TC<int> tc(p);
}

#pragma clang unsafe_buffer_usage begin
struct OptOut {
  OptOut(int *p) : ptr(p) {}
  [[clang::unsafe_buffer_usage]] int *ptr;
};

void test_opt_out(int *p) {
  S a = {0, p};
  S b(0, p);
}
#pragma clang unsafe_buffer_usage end
