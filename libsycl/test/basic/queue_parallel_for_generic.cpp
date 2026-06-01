// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>
#include <iostream>
#include <type_traits>

int main() {
  // TODO: uncomment property once it is implemented. now all sycl::queue
  // objects are in-order due to liboffload limitation. Test is intended to
  // check in-order execution.
  sycl::queue Q{/*sycl::property::queue::in_order()*/};
  auto Dev = Q.get_device();
  auto Ctx = Q.get_context();
  constexpr int N = 8;

  auto A = static_cast<int *>(sycl::malloc_shared(N * sizeof(int), Dev, Ctx));

  for (int i = 0; i < N; ++i) {
    A[i] = 1;
  }

  Q.parallel_for<class IntRange>(N, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
    A[i]++;
  });

  Q.parallel_for<class InitRange>({N}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<1>>::value,
                  "lambda arg type is unexpected");
    A[i]++;
  });

  Q.parallel_for<class InitRange2D>({4, 2}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<2>>::value,
                  "lambda arg type is unexpected");
    A[i.get_linear_id()]++;
  });

  Q.parallel_for<class InitRange3D>({2, 2, 2}, [=](auto i) {
    static_assert(std::is_same<decltype(i), sycl::item<3>>::value,
                  "lambda arg type is unexpected");
    A[i.get_linear_id()]++;
  });

  // TODO: add kernel with offset and kernel with nd_range once they
  // are implemented.

  Q.wait();

  bool Fail{};
  for (int i = 0; i < N; i++) {
    Fail |= !(A[i] == 5);
  }
  sycl::free(A, Ctx);
  return Fail;
}
