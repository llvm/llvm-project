// RUN: %clang_cc1 -fsyntax-only -verify -Wall %s

namespace test1 {
static void f() {} // expected-warning {{function 'f' is not needed and will not be emitted}}
static void f();
template <typename T>
void foo() {
  f();
}
}

namespace test1_template {
template <typename T> static void f() {}
template <> void f<int>() {} // expected-warning {{function 'f<int>' is not needed and will not be emitted}}
template <typename T>
void foo() {
  f<int>();
  f<long>();
}
} // namespace test1_template

namespace test2 {
  static void f() {}
  static void f();
  static void g() { f(); }
  void h() { g(); }
}

namespace test3 {
  static void f();
  template<typename T>
  static void g() {
    f();
  }
  static void f() {
  }
  void h() {
    g<int>();
  }
}

namespace test4 {
  static void f();
  static void f();
  template<typename T>
  static void g() {
    f();
  }
  static void f() {
  }
  void h() {
    g<int>();
  }
}

namespace test4 {
  static void func();
  void bar() {
    void func();
    func();
  }
  static void func() {}
}
