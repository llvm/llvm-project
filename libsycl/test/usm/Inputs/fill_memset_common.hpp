//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>

#include <cassert>
#include <cstddef>

constexpr std::size_t ElementCount = 1024;

template <typename DataT, typename PatternT>
bool verify(DataT *Ptr, PatternT Pattern) {
  for (std::size_t I = 0; I < ElementCount; ++I)
    if (Ptr[I] != static_cast<DataT>(Pattern))
      return false;
  return true;
}

// This function takes ownership of Ptr and is responsible for freeing it.
template <bool VerifyOnDevice, typename DataT, typename OpT, typename PatternT>
void test(sycl::queue &Q, DataT *Ptr, OpT Op, PatternT Pattern) {
  Op(Ptr, Pattern);
  Q.wait();

  if constexpr (VerifyOnDevice) {
    bool *Result = sycl::malloc_shared<bool>(1, Q);
    Q.single_task<class Verify>([=]() { *Result = verify(Ptr, Pattern); });
    Q.wait();
    assert(*Result);
    sycl::free(Result, Q);
  } else {
    assert(verify(Ptr, Pattern));
  }
  sycl::free(Ptr, Q);
}

template <typename DataT, typename OpT, typename PatternT = int>
void runTests(sycl::queue &Q, OpT Op, PatternT Pattern = 42) {
  test<false>(Q, sycl::malloc_host<DataT>(ElementCount, Q), Op, Pattern);
  test<true>(Q, sycl::malloc_host<DataT>(ElementCount, Q), Op, Pattern);
  test<false>(Q, sycl::malloc_shared<DataT>(ElementCount, Q), Op, Pattern);
  test<true>(Q, sycl::malloc_shared<DataT>(ElementCount, Q), Op, Pattern);
  test<true>(Q, sycl::malloc_device<DataT>(ElementCount, Q), Op, Pattern);
}
