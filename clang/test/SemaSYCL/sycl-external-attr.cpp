// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++20 -fsyntax-only -verify -DCPP20 %s
// Semantic tests for sycl_external attribute

[[clang::sycl_external]] // expected-error {{'sycl_external' can only be applied to functions with external linkage}}
static void func1() {}

namespace {
  [[clang::sycl_external]] // expected-error {{'sycl_external' can only be applied to functions with external linkage}}
  void func2() {}

  struct UnnX {};
}

[[clang::sycl_external]] // expected-error {{'sycl_external' can only be applied to functions with external linkage}}
  void func4(UnnX) {}

// The first declaration of a SYCL external function is required to have this attribute.
int foo(); // expected-note {{previous declaration is here}}

[[clang::sycl_external]] int foo(); // expected-error {{'sycl_external' must be applied to the first declaration}}

// Subsequent declrations of a SYCL external function may optionally specify this attribute.
[[clang::sycl_external]] int boo();

[[clang::sycl_external]] int boo(); // OK

int boo(); // OK

class C {
  [[clang::sycl_external]] void member();
};

[[clang::sycl_external]] int main() // expected-error {{'sycl_external' cannot be applied to the 'main' function}}
{
  return 0;
}

class D {
  [[clang::sycl_external]] void del() = delete; // expected-error {{'sycl_external' cannot be applied to an explicitly deleted function}}
};

struct NonCopyable {
  ~NonCopyable() = delete;
  [[clang::sycl_external]] NonCopyable(const NonCopyable&) = default;
};

class A {
  [[clang::sycl_external]]
  A() {}

  [[clang::sycl_external]] void func3() {}
};

class B {
public:
  [[clang::sycl_external]] virtual void foo() {}

  [[clang::sycl_external]] virtual void bar() = 0;
};

[[clang::sycl_external]] int *func0() { return nullptr; }

[[clang::sycl_external]] void func2(int *) {}

[[clang::sycl_external]] constexpr int square(int x);

#ifdef CPP20
[[clang::sycl_external]] consteval int func();
#endif
