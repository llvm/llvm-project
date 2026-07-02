// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

int main() {
  sycl::queue q;

  {
    constexpr size_t G0 = 4;
    constexpr size_t G1 = 6;
    constexpr size_t L0 = 2;
    constexpr size_t L1 = 3;
    int *data = sycl::malloc_shared<int>(G0 * G1, q);
    assert(data);

    for (size_t i = 0; i < G0 * G1; ++i)
      data[i] = -1;

    q.submit([&](sycl::handler &cgh) {
       cgh.parallel_for<class HandlerNDRange2DRuntime>(
           sycl::nd_range<2>{sycl::range<2>{G0, G1}, sycl::range<2>{L0, L1}},
           [=](sycl::nd_item<2> it) {
             const size_t i = it.get_global_id(0);
             const size_t j = it.get_global_id(1);
             data[i * G1 + j] = static_cast<int>(i * 100 + j);
           });
     }).wait();

    for (size_t i = 0; i < G0; ++i)
      for (size_t j = 0; j < G1; ++j)
        assert(data[i * G1 + j] == static_cast<int>(i * 100 + j));

    sycl::free(data, q);
  }

  {
    constexpr size_t G0 = 2;
    constexpr size_t G1 = 3;
    constexpr size_t G2 = 4;
    constexpr size_t L0 = 1;
    constexpr size_t L1 = 3;
    constexpr size_t L2 = 2;
    int *data = sycl::malloc_shared<int>(G0 * G1 * G2, q);
    assert(data);

    for (size_t i = 0; i < G0 * G1 * G2; ++i)
      data[i] = -1;

    q.submit([&](sycl::handler &cgh) {
       cgh.parallel_for<class HandlerNDRange3DRuntime>(
           sycl::nd_range<3>{sycl::range<3>{G0, G1, G2},
                             sycl::range<3>{L0, L1, L2}},
           [=](sycl::nd_item<3> it) {
             const size_t i = it.get_global_id(0);
             const size_t j = it.get_global_id(1);
             const size_t k = it.get_global_id(2);
             data[(i * G1 + j) * G2 + k] =
                 static_cast<int>(i * 100 + j * 10 + k);
           });
     }).wait();

    for (size_t i = 0; i < G0; ++i)
      for (size_t j = 0; j < G1; ++j)
        for (size_t k = 0; k < G2; ++k)
          assert(data[(i * G1 + j) * G2 + k] ==
                 static_cast<int>(i * 100 + j * 10 + k));

    sycl::free(data, q);
  }

  return 0;
}
