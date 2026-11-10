//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cassert>

void test1D(sycl::queue &Q) {
  constexpr size_t N = 16;
  constexpr size_t LocalSize = 4;
  int *Data = sycl::malloc_shared<int>(N, Q);
  assert(Data);

  for (size_t I = 0; I < N; ++I)
    Data[I] = 0;

  Q.submit([&](sycl::handler &CGH) {
     CGH.parallel_for<class HandlerParallelForRuntime>(
         sycl::range<1>{N},
         [=](sycl::item<1> It) { Data[It[0]] = static_cast<int>(It[0]) + 7; });
   }).wait();

  for (size_t I = 0; I < N; ++I)
    assert(Data[I] == static_cast<int>(I) + 7);

  for (size_t I = 0; I < N; ++I)
    Data[I] = 0;

  Q.submit([&](sycl::handler &CGH) {
     CGH.parallel_for<class HandlerParallelForNDRangeRuntime>(
         sycl::nd_range<1>{sycl::range<1>{N}, sycl::range<1>{LocalSize}},
         [=](sycl::nd_item<1> It) {
           const size_t Idx = It.get_global_id(0);
           Data[Idx] = static_cast<int>(Idx) + 11;
         });
   }).wait();

  for (size_t I = 0; I < N; ++I)
    assert(Data[I] == static_cast<int>(I) + 11);

  sycl::free(Data, Q);
}

void test2D(sycl::queue &Q) {
  constexpr size_t G0 = 4;
  constexpr size_t G1 = 6;
  constexpr size_t L0 = 2;
  constexpr size_t L1 = 3;
  int *Data = sycl::malloc_shared<int>(G0 * G1, Q);
  assert(Data);

  for (size_t I = 0; I < G0 * G1; ++I)
    Data[I] = -1;

  Q.submit([&](sycl::handler &CGH) {
     CGH.parallel_for<class HandlerParallelFor2DRuntime>(
         sycl::range<2>{G0, G1}, [=](sycl::item<2> It) {
           const size_t I = It.get_id(0);
           const size_t J = It.get_id(1);
           Data[I * G1 + J] = static_cast<int>(I * 100 + J) + 7;
         });
   }).wait();

  for (size_t I = 0; I < G0; ++I)
    for (size_t J = 0; J < G1; ++J)
      assert(Data[I * G1 + J] == static_cast<int>(I * 100 + J) + 7);

  for (size_t I = 0; I < G0 * G1; ++I)
    Data[I] = -1;

  Q.submit([&](sycl::handler &CGH) {
     CGH.parallel_for<class HandlerNDRange2DRuntime>(
         sycl::nd_range<2>{sycl::range<2>{G0, G1}, sycl::range<2>{L0, L1}},
         [=](sycl::nd_item<2> It) {
           const size_t I = It.get_global_id(0);
           const size_t J = It.get_global_id(1);
           Data[I * G1 + J] = static_cast<int>(I * 100 + J);
         });
   }).wait();

  for (size_t I = 0; I < G0; ++I)
    for (size_t J = 0; J < G1; ++J)
      assert(Data[I * G1 + J] == static_cast<int>(I * 100 + J));

  sycl::free(Data, Q);
}

void test3D(sycl::queue &Q) {
  constexpr size_t G0 = 2;
  constexpr size_t G1 = 3;
  constexpr size_t G2 = 4;
  constexpr size_t L0 = 1;
  constexpr size_t L1 = 3;
  constexpr size_t L2 = 2;
  int *Data = sycl::malloc_shared<int>(G0 * G1 * G2, Q);
  assert(Data);

  for (size_t I = 0; I < G0 * G1 * G2; ++I)
    Data[I] = -1;

  Q.submit([&](sycl::handler &CGH) {
     CGH.parallel_for<class HandlerParallelFor3DRuntime>(
         sycl::range<3>{G0, G1, G2}, [=](sycl::item<3> It) {
           const size_t I = It.get_id(0);
           const size_t J = It.get_id(1);
           const size_t K = It.get_id(2);
           Data[(I * G1 + J) * G2 + K] =
               static_cast<int>(I * 100 + J * 10 + K) + 7;
         });
   }).wait();

  for (size_t I = 0; I < G0; ++I)
    for (size_t J = 0; J < G1; ++J)
      for (size_t K = 0; K < G2; ++K)
        assert(Data[(I * G1 + J) * G2 + K] ==
               static_cast<int>(I * 100 + J * 10 + K) + 7);

  for (size_t I = 0; I < G0 * G1 * G2; ++I)
    Data[I] = -1;

  Q.submit([&](sycl::handler &CGH) {
     CGH.parallel_for<class HandlerNDRange3DRuntime>(
         sycl::nd_range<3>{sycl::range<3>{G0, G1, G2},
                           sycl::range<3>{L0, L1, L2}},
         [=](sycl::nd_item<3> It) {
           const size_t I = It.get_global_id(0);
           const size_t J = It.get_global_id(1);
           const size_t K = It.get_global_id(2);
           Data[(I * G1 + J) * G2 + K] = static_cast<int>(I * 100 + J * 10 + K);
         });
   }).wait();

  for (size_t I = 0; I < G0; ++I)
    for (size_t J = 0; J < G1; ++J)
      for (size_t K = 0; K < G2; ++K)
        assert(Data[(I * G1 + J) * G2 + K] ==
               static_cast<int>(I * 100 + J * 10 + K));

  sycl::free(Data, Q);
}

int main() {
  sycl::queue Q;
  test1D(Q);
  test2D(Q);
  test3D(Q);
  return 0;
}
