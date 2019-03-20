// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s
// expected-no-diagnostics

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}
using bool1 = bool;
using bool2 = bool __attribute__((ext_vector_type(2)));
using bool3 = bool __attribute__((ext_vector_type(3)));
using bool4 = bool __attribute__((ext_vector_type(4)));
using bool8 = bool __attribute__((ext_vector_type(8)));
using bool16 = bool __attribute__((ext_vector_type(16)));

int main() {
  kernel<class kernel_function>(
      [=]() {
        bool1 b1;
        bool2 b2;
        bool3 b3;
        bool4 b4;
        bool8 b8;
        bool16 b16;
      });
  return 0;
}

