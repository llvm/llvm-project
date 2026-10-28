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
#include <cstddef>

using namespace sycl;

constexpr std::size_t Count = 1024;
constexpr std::size_t NumBytes = Count * sizeof(int);

int main() {

  sycl::queue Q;
  sycl::usm_allocator<uint32_t, usm::alloc::host> Alloc(Q);

  std::vector<uint32_t, decltype(Alloc)> Vec;

  return 0;
}
