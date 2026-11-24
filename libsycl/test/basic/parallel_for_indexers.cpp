//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: any-device
// RUN: %clangxx -fsycl -Wno-error=deprecated-declarations %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

// TODO: original test works with buffers, revert changes to USM once they are
// implemented.
int main() {
  bool Fail{};

  constexpr size_t DataSize = 10;
  const range<1> GlobalRange(6);
  // Id indexer
  {
    queue Q;
    int *Data = sycl::malloc_shared<int>(DataSize, Q);
    for (size_t I = 0; I < DataSize; ++I)
      Data[I] = -1;

    Q.parallel_for<class Id1>(GlobalRange,
                              [=](id<1> Index) { Data[Index] = Index[0]; });
    Q.wait();

    Fail |= [&]() {
      for (size_t I = 0; I < DataSize; ++I) {
        const int ExpectedVal = I < GlobalRange[0] ? I : -1;
        if (Data[I] != ExpectedVal) {
          std::cout << "line: " << __LINE__ << " Data[" << I << "] is "
                    << Data[I] << " expected " << ExpectedVal << std::endl;
          return true;
        }
      }
      return false;
    }();

    free(Data, Q);
  }

  // Item indexer without offset
  {
    // TODO: replace struct with sycl::int2 once implemented.
    struct DoubleInt {
      int Id;
      int Range;
    };
    queue Q;
    DoubleInt *Data = sycl::malloc_shared<DoubleInt>(DataSize, Q);
    for (size_t I = 0; I < DataSize; ++I)
      Data[I] = {-1, -1};

    Q.parallel_for<class Item1NoOffset>(GlobalRange, [=](item<1, false> Index) {
      Data[Index.get_id()] = {int(Index.get_id()[0]),
                              int(Index.get_range()[0])};
    });
    Q.wait();

    Fail |= [&]() {
      for (size_t I = 0; I < DataSize; ++I) {
        const int ExpectedValID = I < GlobalRange[0] ? I : -1;
        const int ExpectedValRange = I < GlobalRange[0] ? GlobalRange[0] : -1;
        if (Data[I].Id != ExpectedValID || Data[I].Range != ExpectedValRange) {
          std::cout << "line: " << __LINE__ << " Data[" << I << "] is {"
                    << Data[I].Id << ", " << Data[I].Range << "} expected {"
                    << ExpectedValID << ", " << ExpectedValRange << "}"
                    << std::endl;
          return true;
        }
      }
      return false;
    }();
    free(Data, Q);
  }

  // TODO: Item indexer with offset, blocked by liboffload support.

  // TODO: add nd_item check
  return Fail;
}
