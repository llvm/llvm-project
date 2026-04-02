// RUN: %clang_cc1 -fsyntax-only -Wunused-but-set-variable -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -Wunused-but-set-variable -Wno-unused-but-set-global -verify=no-global -std=c++17 %s
// no-global-no-diagnostics

static thread_local int tl_set_unused;  // expected-warning {{variable 'tl_set_unused' set but not used}}
static thread_local int tl_set_and_used;
thread_local int tl_no_static_set_unused;

// Warning should respect attributes.
[[maybe_unused]] static int with_maybe_unused;
__attribute__((unused)) static int with_unused_attr;

void f0() {
  tl_set_unused = 1;
  tl_set_and_used = 2;
  int x = tl_set_and_used;
  (void)x;

  tl_no_static_set_unused = 3;

  with_maybe_unused = 4;
  with_unused_attr = 5;
}

// Named namespace.
namespace test {
  static int set_unused;  // expected-warning {{variable 'set_unused' set but not used}}
  static int set_and_used;

  void f1() {
    set_unused = 1;
    set_and_used = 2;
    int x = set_and_used;
    (void)x;
  }
}

// Nested named namespace.
namespace outer {
namespace inner {
static int nested_unused; // expected-warning {{variable 'nested_unused' set but not used}}
void f2() {
  nested_unused = 5;
}
}
}

// Anonymous namespace (all vars have internal linkage).
namespace {
  int anon_ns_unused; // expected-warning {{variable 'anon_ns_unused' set but not used}}
  static int anon_ns_static_unused; // expected-warning {{variable 'anon_ns_static_unused' set but not used}}

  void f3() {
    anon_ns_unused = 1;
    anon_ns_static_unused = 2;
  }
}

// Function pointers at file scope.
static void (*unused_func_ptr)(); // expected-warning {{variable 'unused_func_ptr' set but not used}}
void SetUnusedCallback(void (*f)()) {
  unused_func_ptr = f;
}

static void (*used_func_ptr)();
void SetUsedCallback(void (*f)()) {
  used_func_ptr = f;
}
void CallUsedCallback() {
  if (used_func_ptr)
    used_func_ptr();
}

// Static data members (external linkage, should not warn).
class MyClass {
public:
  static int unused_static_member;
  static int used_static_member;
};

int MyClass::unused_static_member = 0;
int MyClass::used_static_member = 0;

void f4() {
  MyClass::unused_static_member = 10;

  MyClass::used_static_member = 20;
  int x = MyClass::used_static_member;
  (void)x;
}

// Static data members in a named namespace (external linkage, should not warn).
namespace named {
  struct NamedClass {
    static int w;
  };
  int NamedClass::w = 0;
}

void f5() {
  named::NamedClass::w = 4;
}

// Static data members in anonymous namespace (internal linkage, should warn).
namespace {
  class AnonClass {
  public:
    static int unused_member; // expected-warning {{variable 'unused_member' set but not used}}
    static int used_member;
  };

  int AnonClass::unused_member = 0;
  int AnonClass::used_member = 0;
}

void f6() {
  AnonClass::unused_member = 3;
  AnonClass::used_member = 4;
  int y = AnonClass::used_member;
  (void)y;
}

// Static data members in nested anonymous namespace (internal linkage, should warn).
namespace outer2 {
  namespace {
    struct NestedAnonClass {
      static int v; // expected-warning {{variable 'v' set but not used}}
    };
    int NestedAnonClass::v = 0;
  }
}

void f7() {
  outer2::NestedAnonClass::v = 5;
}

// Static data members set inside methods, read outside.
namespace {
  struct SetInMethod {
    static int x;
    static int y; // expected-warning {{variable 'y' set but not used}}
    void setX() { x = 1; }
    void setY() { y = 1; }
  };
  int SetInMethod::x;
  int SetInMethod::y;
}

void f8() {
  SetInMethod s;
  s.setX();
  s.setY();
  int v = SetInMethod::x;
  (void)v;  // only x is read
}

// External linkage static data members set inside methods.
struct ExtSetInMethod {
  static int x;
  void set() { x = 1; }
};
int ExtSetInMethod::x;

void f9() {
  ExtSetInMethod e;
  e.set();
}
