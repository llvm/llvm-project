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
#include <cstdint>
#include <vector>

using namespace sycl;

constexpr size_t Count = 1024;
constexpr size_t Align = 256;

struct alignas(Align) OverAligned {
  int Value;
};

template <usm::alloc Kind> void testVector(queue &Q) {
  usm_allocator<int, Kind> Alloc(Q);
  std::vector<int, decltype(Alloc)> Vec(Alloc);
  for (size_t I = 0; I < Count; ++I)
    Vec.push_back(static_cast<int>(I));

  int *Data = Vec.data();
  Q.parallel_for(Count, [=](id<1> I) { Data[I] *= 2; }).wait();

  for (size_t I = 0; I < Count; ++I)
    assert(Vec[I] == static_cast<int>(2 * I));
}

template <typename AllocT> void testAlignment(AllocT Alloc) {
  // The first allocation might be over-aligned by chance, so check several.
  constexpr size_t NumPtrs = 10;
  typename AllocT::value_type *Ptrs[NumPtrs];
  for (size_t I = 0; I < NumPtrs; ++I)
    Ptrs[I] = Alloc.allocate(1);
  for (size_t I = 0; I < NumPtrs; ++I)
    assert(reinterpret_cast<uintptr_t>(Ptrs[I]) % Align == 0);
  for (size_t I = 0; I < NumPtrs; ++I)
    Alloc.deallocate(Ptrs[I], 1);
}

template <usm::alloc Kind> void test(queue &Q) {
  testVector<Kind>(Q);
  testAlignment(usm_allocator<OverAligned, Kind>(Q));
  testAlignment(usm_allocator<int, Kind, Align>(Q));
}

int main() {
  queue Q;
  device Dev = Q.get_device();

  if (Dev.has(aspect::usm_host_allocations))
    test<usm::alloc::host>(Q);
  if (Dev.has(aspect::usm_shared_allocations))
    test<usm::alloc::shared>(Q);

  return 0;
}
