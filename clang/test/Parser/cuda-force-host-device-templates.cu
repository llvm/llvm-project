// RUN: %clang_cc1 -std=c++14 -S -verify -fcuda-is-device %s -o /dev/null

// Check how the force_cuda_host_device pragma interacts with template
// instantiations.  The errors here are emitted at codegen, so we can't do
// -fsyntax-only.

template <typename T>
auto foo() {  // expected-note {{declared here}}
  return T();
}

template <typename T>
struct X {
  void foo(); // expected-note {{declared here}}
};

#pragma clang force_cuda_host_device begin
__attribute__((host)) __attribute__((device)) void test() {
  int n = foo<int>();  // expected-error {{reference to __host__ function 'foo<int>'}}
  X<int>().foo();  // expected-error {{reference to __host__ function 'foo'}}
}
#pragma clang force_cuda_host_device end

// Same thing as above, but within a force_cuda_host_device block without a
// corresponding end.

template <typename T>
T bar() {  // expected-note {{declared here}}
  return T();
}

template <typename T>
struct Y {
  void bar(); // expected-note {{declared here}}
};

#pragma clang force_cuda_host_device begin
__attribute__((host)) __attribute__((device)) void test2() {
  int n = bar<int>();  // expected-error {{reference to __host__ function 'bar<int>'}}
  Y<int>().bar();  // expected-error {{reference to __host__ function 'bar'}}
}
