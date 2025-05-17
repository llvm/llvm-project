// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -DSYCL %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify -DHOST %s
// RUN: %clang_cc1 -verify %s

// Semantic tests for sycl_external attribute

#ifdef SYCL

__attribute__((sycl_external(3))) // expected-error {{'sycl_external' attribute takes no arguments}}
void bar() {}

__attribute__((sycl_external)) // expected-error {{'sycl_external' attribute cannot be applied to a function without external linkage}}
static void func1() {}

namespace {
  __attribute__((sycl_external)) // expected-error {{'sycl_external' attribute cannot be applied to a function without external linkage}}
  void func2() {}

  struct UnnX {};
}

__attribute__((sycl_external)) // expected-error {{'sycl_external' attribute cannot be applied to a function without external linkage}}
  void func4(UnnX) {}

class A {
  __attribute__((sycl_external))
  A() {}

  __attribute__((sycl_external)) void func3() {}
};

class B {
public:
  __attribute__((sycl_external)) virtual void foo() {}

  __attribute__((sycl_external)) virtual void bar() = 0;
};

__attribute__((sycl_external)) int *func0() { return nullptr; }

__attribute__((sycl_external)) void func2(int *) {}

#elif defined(HOST)

// expected-no-diagnostics
__attribute__((sycl_external)) void func3() {}

#else
__attribute__((sycl_external)) // expected-warning {{'sycl_external' attribute ignored}}
void baz() {}

#endif
