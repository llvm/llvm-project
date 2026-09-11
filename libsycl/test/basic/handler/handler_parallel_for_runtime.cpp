// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

void test1D(sycl::queue &q) {
  constexpr size_t N = 16;
  constexpr size_t LocalSize = 4;
  int *data = sycl::malloc_shared<int>(N, q);
  assert(data);

  for (size_t i = 0; i < N; ++i)
    data[i] = 0;

  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for<class HandlerParallelForRuntime>(
         sycl::range<1>{N},
         [=](sycl::item<1> it) { data[it[0]] = static_cast<int>(it[0]) + 7; });
   }).wait();

  for (size_t i = 0; i < N; ++i)
    assert(data[i] == static_cast<int>(i) + 7);

  for (size_t i = 0; i < N; ++i)
    data[i] = 0;

  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for<class HandlerParallelForNDRangeRuntime>(
         sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{LocalSize}},
         [=](sycl::nd_item<1> it) {
           const size_t idx = it.get_global_id(0);
           data[idx] = static_cast<int>(idx) + 11;
         });
   }).wait();

  for (size_t i = 0; i < N; ++i)
    assert(data[i] == static_cast<int>(i) + 11);

  sycl::free(data, q);
}

void test2D(sycl::queue &q) {
  constexpr size_t G0 = 4;
  constexpr size_t G1 = 6;
  constexpr size_t L0 = 2;
  constexpr size_t L1 = 3;
  int *data = sycl::malloc_shared<int>(G0 * G1, q);
  assert(data);

  for (size_t i = 0; i < G0 * G1; ++i)
    data[i] = -1;

  q.submit([&](sycl::handler &cgh) {
     cgh.parallel_for<class HandlerParallelFor2DRuntime>(
         sycl::range<2>{G0, G1}, [=](sycl::item<2> it) {
           const size_t i = it.get_id(0);
           const size_t j = it.get_id(1);
           data[i * G1 + j] = static_cast<int>(i * 100 + j) + 7;
         });
   }).wait();

  for (size_t i = 0; i < G0; ++i)
    for (size_t j = 0; j < G1; ++j)
      assert(data[i * G1 + j] == static_cast<int>(i * 100 + j) + 7);

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

void test3D(sycl::queue &q) {
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
     cgh.parallel_for<class HandlerParallelFor3DRuntime>(
         sycl::range<3>{G0, G1, G2}, [=](sycl::item<3> it) {
           const size_t i = it.get_id(0);
           const size_t j = it.get_id(1);
           const size_t k = it.get_id(2);
           data[(i * G1 + j) * G2 + k] =
               static_cast<int>(i * 100 + j * 10 + k) + 7;
         });
   }).wait();

  for (size_t i = 0; i < G0; ++i)
    for (size_t j = 0; j < G1; ++j)
      for (size_t k = 0; k < G2; ++k)
        assert(data[(i * G1 + j) * G2 + k] ==
               static_cast<int>(i * 100 + j * 10 + k) + 7);

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
           data[(i * G1 + j) * G2 + k] = static_cast<int>(i * 100 + j * 10 + k);
         });
   }).wait();

  for (size_t i = 0; i < G0; ++i)
    for (size_t j = 0; j < G1; ++j)
      for (size_t k = 0; k < G2; ++k)
        assert(data[(i * G1 + j) * G2 + k] ==
               static_cast<int>(i * 100 + j * 10 + k));

  sycl::free(data, q);
}

int main() {
  sycl::queue q;
  test1D(q);
  test2D(q);
  test3D(q);
  return 0;
}
