// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

int sink(int);

namespace std { enum class byte : unsigned char {}; }

struct Trivial { int m; };
struct WithCtor { WithCtor(); int m; };
struct ByteWrap { std::byte b; };
struct ByteAndInt { std::byte b; int x; };

void test_byte() {
  // std::byte may be left uninitialized (paper section 4), as may arrays of it
  // and records whose only members are std::byte.
  std::byte a;
  std::byte buf[8];
  ByteWrap w;
  ByteAndInt m;       // expected-error {{variable 'm' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  (void)a; (void)buf; (void)w; (void)m;
}

void test_scalars() {
  int a;            // expected-error {{variable 'a' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  int b = 0;
  int c [[uninit]];
  int d{};
  int e = sink(1);
  (void)b; (void)c; (void)d; (void)e;
}

void test_pointer() {
  int* p;           // expected-error {{variable 'p' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  int* q = nullptr;
  // A pointer cannot be left uninitialized (paper section 4.1); the marker is
  // rejected rather than excusing it.
  int* r [[uninit]]; // expected-error {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}
  (void)q; (void)r;
}

struct PtrMember {
  int* p [[uninit]]; // expected-error {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}
};

void test_pointer_marker_suppressed() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] int* p [[uninit]];
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "pointer_marker")]] int* q [[uninit]];
  (void)p; (void)q;
}

enum E { E0, E1 };
void test_enum() {
  E x;              // expected-error {{variable 'x' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  E y = E0;
  E z [[uninit]];
  (void)y; (void)z;
}

void test_class_with_user_ctor() {
  // OK: user-provided default constructor is trusted. The marker is
  // unnecessary on class types whose default-init runs a constructor; in
  // fact combining them is a contradiction caught by R4 (the constructor
  // call is the synthesized initializer).
  WithCtor w;
  (void)w;
}

void test_class_trivial() {
  // A trivial / aggregate class whose default-initialization leaves a scalar
  // member indeterminate is diagnosed by R5 (uninit_decl), the §6
  // "classes without constructors" rule. (Detailed coverage lives in
  // safety-profile-init-aggregate.cpp.)
  Trivial t; // expected-error {{variable 't' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  (void)t;
}

void test_static_local() {
  static int s;
  thread_local int t;
  (void)s; (void)t;
}

int g_namespace_scalar;
thread_local int g_thread;
extern int g_extern;

void test_param(int p) {
  (void)p;
}

void test_marker_then_assign() {
  int x [[uninit]];
  x = 7;
  (void)x;
}

void test_suppress_decl() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "uninit_decl")]] int x;
  (void)x;
}

void test_suppress_block() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] {
    int a;
    int b;
    (void)a; (void)b;
  }
}

template <typename T>
void template_uninit() {
  T x;              // expected-error {{variable 'x' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  (void)x;
}
template void template_uninit<int>(); // expected-note {{in instantiation of function template specialization 'template_uninit<int>' requested here}}
