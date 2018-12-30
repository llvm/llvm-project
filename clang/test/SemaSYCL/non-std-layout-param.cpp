// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -std=c++11 %s

// This test checks if compiler reports compilation error on an attempt to pass
// non-standard layout struct object as SYCL kernel parameter.

struct Base {
  int X;
};

// This struct has non-standard layout, because both C (the most derived class)
// and Base have non-static data members.
struct C : public Base {
  int Y;
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}


void test() {
  // expected-error@+1 2{{kernel parameter has non-standard layout class/struct type}}
  C C0;
  C0.Y=0;
  kernel_single_task<class MyKernel>([&] { C0.Y++; });
}

