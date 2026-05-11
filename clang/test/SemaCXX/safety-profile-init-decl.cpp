// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(std::init)]];

int sink(int);

struct Trivial { int m; };
struct WithCtor { WithCtor(); int m; };

void test_scalars() {
  int a;            // expected-error {{variable 'a' must be initialized or marked '[[uninitialized]]' under profile 'std::init'}}
  int b = 0;
  int c [[uninitialized]];
  int d{};
  int e = sink(1);
  (void)b; (void)c; (void)d; (void)e;
}

void test_pointer() {
  int* p;           // expected-error {{variable 'p' must be initialized or marked '[[uninitialized]]' under profile 'std::init'}}
  int* q = nullptr;
  int* r [[uninitialized]];
  (void)q; (void)r;
}

enum E { E0, E1 };
void test_enum() {
  E x;              // expected-error {{variable 'x' must be initialized or marked '[[uninitialized]]' under profile 'std::init'}}
  E y = E0;
  E z [[uninitialized]];
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
  // Conservative: trivial / aggregate class types are not diagnosed by R2 in
  // this minimal slice. Field-level initialization tracking is the §6
  // "classes without constructors" work and is explicitly deferred.
  Trivial t;
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
  int x [[uninitialized]];
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
  T x;              // expected-error {{variable 'x' must be initialized or marked '[[uninitialized]]' under profile 'std::init'}}
  (void)x;
}
template void template_uninit<int>(); // expected-note {{in instantiation of function template specialization 'template_uninit<int>' requested here}}
