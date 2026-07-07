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

// The marker checks key on the base element type: an array of pointers is
// banned exactly like a single pointer (paper section 4.1) -- the marker must
// not smuggle uninitialized pointers past uninit_decl element-wise.
void test_pointer_array() {
  [[uninit]] int* a[2];    // expected-error {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}
  [[uninit]] int* b[2][3]; // expected-error {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}
  (void)a; (void)b;
}

struct PtrArrayMember {
  [[uninit]] int* a[2]; // expected-error {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}
};

void test_pointer_array_marker_suppressed() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init, rule: "pointer_marker")]] [[uninit]] int* a[2];
  (void)a;
}

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

// A *non-dependent* uninitialized local in a template body is diagnosed once --
// at instantiation -- not on the template pattern (no double-fire).
template <typename T>
void template_uninit_nondependent() {
  int x;            // expected-error {{variable 'x' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  (void)x;
}
template void template_uninit_nondependent<int>(); // expected-note {{in instantiation of function template specialization 'template_uninit_nondependent<int>' requested here}}

// An uninstantiated template pattern never reaches phase 7, so no rule fires.
template <typename T>
void template_uninit_never_instantiated() {
  int x;
  (void)x;
}

// A dependent local that substitutes to a pointer is deferred on the pattern
// and fires pointer_marker at instantiation, not on the template.
template <typename T>
void template_ptr_marker() {
  T x [[uninit]]; // #template-ptr-marker
  (void)x;
}
template void template_ptr_marker<int *>(); // expected-note {{in instantiation of function template specialization 'template_ptr_marker<int *>' requested here}}
// expected-error@#template-ptr-marker {{'[[uninit]]' cannot be applied to a pointer under profile 'std::init'; initialize the pointer (for example to 'nullptr')}}

// A [[profiles::suppress(std::init)]] live at the point of instantiation
// covers the trigger's tokens, not the pattern's (P3589R2 s2.4p3): rules
// inside a synchronously instantiated pattern must still fire.
template <typename T>
auto instantiation_leak_use() {
  T t; // expected-error {{variable 't' must be initialized or marked '[[uninit]]' under profile 'std::init'}}
  (void)t;
  return 0;
}
void instantiation_leak_trigger() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(std::init)]] int leaked = instantiation_leak_use<int>(); // expected-note {{in instantiation of function template specialization 'instantiation_leak_use<int>' requested here}}
  (void)leaked;
}

// A declarator-position suppress covers a diagnostic located at the declared
// name: entries anchor to the construct's begin, not the attribute's.
void declarator_position_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  int c [[profiles::suppress(std::init, rule: "uninit_decl")]];
  (void)c;
}
