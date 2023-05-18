// RUN: %clang_cc1 -std=c++1y -triple i386-pc-win32 -fms-compatibility -fms-extensions -fsyntax-only -verify %s
// expected-no-diagnostics

int foo() {
  static_assert(sizeof(__FUNCDNAME__) == 12, "?foo@@YAHXZ");
  return 0;
}

// Within templates.
template <typename T>
int baz() {
  static_assert(sizeof(__FUNCDNAME__) == 16, "??$baz@H@@YAHXZ");

  return 0;
}

struct A {
  A() {
    static_assert(sizeof(__FUNCDNAME__) == 13, "??0A@@QAE@XZ");
  }
  ~A() {
    static_assert(sizeof(__FUNCDNAME__) == 13, "??1A@@QAE@XZ");
  }
};

int main() {
  static_assert(sizeof(__FUNCDNAME__) == 5, "main");

  baz<int>();

  return 0;
}
