// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

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

// FIXME: The first declaration of a function is required to have the attribute.
// The attribute may be optionally present on subsequent declarations
int foo(int c);

[[clang::sycl_external]] void foo();

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

