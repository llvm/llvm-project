// RUN: %clangxx -fsycl -fsyntax-only %s
// expected-no-diagnostics

// Adapted from upstream SYCL test/basic_tests/handler/
// parallel_for_arg_restrictions.cpp. This version keeps signatures that
// should compile for current handler::parallel_for(range) support.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range{1}, [=](sycl::item<1>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range{1, 1}, [=](sycl::item<2>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range{1, 1, 1}, [=](sycl::item<3>) {});
  });

  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(sycl::range{1}, [=](sycl::id<1>) {});
  });

  return 0;
}
