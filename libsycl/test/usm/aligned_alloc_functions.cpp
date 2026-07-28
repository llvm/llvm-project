// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

void check_aligned_allocation(size_t Alignment, auto AllocFn, queue &q) {
  constexpr size_t N = 10;
  void *ptrs[N];

  // 1. Allocate multiple blocks
  for (size_t i = 0; i < N; ++i) {
    ptrs[i] = AllocFn();
    assert(ptrs[i] != nullptr && "Allocation returned nullptr!");

    // 2. Bitwise alignment check
    auto addr = reinterpret_cast<uintptr_t>(ptrs[i]);
    if ((addr & (Alignment - 1)) != 0) {
      std::cerr << "Address " << ptrs[i] << " not aligned to " << Alignment
                << " bytes!\n";
      assert(false && "Alignment check failed");
    }
  }

  // 3. Cleanup
  for (size_t i = 0; i < N; ++i) {
    free(ptrs[i], q);
  }
}

int main() {
  queue q;
  context ctx = q.get_context();
  device d = q.get_device();

  size_t Alignments[] = {16, 32, 64, 128, 256, 512};
  size_t Size = 1024;

  for (size_t Align : Alignments) {
    // Test aligned_alloc_device
    check_aligned_allocation(
        Align, [&]() { return aligned_alloc_device(Align, Size, q); }, q);
    check_aligned_allocation(
        Align, [&]() { return aligned_alloc_device(Align, Size, d, ctx); }, q);

    // Test aligned_alloc_host
    if (d.has(aspect::usm_host_allocations)) {
      check_aligned_allocation(
          Align, [&]() { return aligned_alloc_host(Align, Size, q); }, q);
      check_aligned_allocation(
          Align, [&]() { return aligned_alloc_host(Align, Size, ctx); }, q);
    }

    // Test aligned_alloc_shared
    if (d.has(aspect::usm_shared_allocations)) {
      check_aligned_allocation(
          Align, [&]() { return aligned_alloc_shared(Align, Size, q); }, q);
      check_aligned_allocation(
          Align, [&]() { return aligned_alloc_shared(Align, Size, d, ctx); },
          q);
    }

    // Test generic aligned_alloc
    check_aligned_allocation(
        Align,
        [&]() { return aligned_alloc(Align, Size, q, usm::alloc::device); }, q);
    check_aligned_allocation(
        Align,
        [&]() {
          return aligned_alloc(Align, Size, d, ctx, usm::alloc::device);
        },
        q);
  }

  std::cout << "All aligned USM E2E tests passed!\n";
  return 0;
}