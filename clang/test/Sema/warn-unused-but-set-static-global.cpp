// RUN: %clang_cc1 -fsyntax-only -Wunused-but-set-variable -verify -std=c++11 %s

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
