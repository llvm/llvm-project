// RUN: %clang_cc1 -fsyntax-only -Wunused-but-set-variable -verify -std=c++11 %s

static thread_local int tl_set_unused;  // expected-warning {{variable 'tl_set_unused' set but not used}}
static thread_local int tl_set_and_used;
thread_local int tl_no_static_set_unused;

void f0() {
  tl_set_unused = 1;
  tl_set_and_used = 2;
  int x = tl_set_and_used;
  (void)x;

  tl_no_static_set_unused = 3;
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

  // Function pointer in namespace.
  static void (*sandboxing_callback)();
  void SetSandboxingCallback(void (*f)()) {
    sandboxing_callback = f;
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
