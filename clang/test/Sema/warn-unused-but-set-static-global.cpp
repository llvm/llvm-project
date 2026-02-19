// RUN: %clang_cc1 -fsyntax-only -Wunused-but-set-variable -verify -std=c++17 %s

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

namespace outer {
namespace inner {
static int nested_unused; // expected-warning {{variable 'nested_unused' set but not used}}
void f2() {
  nested_unused = 5;
}
}
}

// Anonymous namespace (all vars have internal linkage)
namespace {
  int anon_ns_unused; // expected-warning {{variable 'anon_ns_unused' set but not used}}
  static int anon_ns_static_unused; // expected-warning {{variable 'anon_ns_static_unused' set but not used}}

  // Should not warn on static data members in current implementation.
  class AnonClass {
  public:
    static int unused_member;
  };

  int AnonClass::unused_member = 0;

  void f3() {
    anon_ns_unused = 1;
    anon_ns_static_unused = 2;
    AnonClass::unused_member = 3;
  }
}

// Function pointers at file scope (unused)
static void (*unused_func_ptr)(); // expected-warning {{variable 'unused_func_ptr' set but not used}}
void SetUnusedCallback(void (*f)()) {
  unused_func_ptr = f;
}

// Function pointers at file scope (used)
static void (*used_func_ptr)();
void SetUsedCallback(void (*f)()) {
  used_func_ptr = f;
}
void CallUsedCallback() {
  if (used_func_ptr)
    used_func_ptr();
}

// Static data members (have external linkage so should not warn).
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
