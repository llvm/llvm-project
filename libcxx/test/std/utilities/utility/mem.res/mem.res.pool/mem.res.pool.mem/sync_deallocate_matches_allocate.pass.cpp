//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: availability-pmr-missing

// <memory_resource>

// class synchronized_pool_resource

#include <memory_resource>
#include <algorithm>
#include <cassert>
#include <new>
#include <vector>

struct allocation_record {
  std::size_t bytes;
  std::size_t align;
  explicit allocation_record(std::size_t b, size_t a) : bytes(b), align(a) {}
  bool operator==(const allocation_record& rhs) const { return (bytes == rhs.bytes) && (align == rhs.align); }
  bool operator<(const allocation_record& rhs) const {
    if (bytes != rhs.bytes)
      return (bytes < rhs.bytes);
    return (align < rhs.align);
  }
};

class test_resource : public std::pmr::memory_resource {
  void* do_allocate(std::size_t bytes, size_t align) override {
    void* result = std::pmr::new_delete_resource()->allocate(bytes, align);
    successful_allocations.emplace_back(bytes, align);
    return result;
  }
  void do_deallocate(void* p, std::size_t bytes, size_t align) override {
    deallocations.emplace_back(bytes, align);
    return std::pmr::new_delete_resource()->deallocate(p, bytes, align);
  }
  bool do_is_equal(const std::pmr::memory_resource&) const noexcept override { return false; }

public:
  std::vector<allocation_record> successful_allocations;
  std::vector<allocation_record> deallocations;
};

template <class F>
void test_allocation_pattern(F do_pattern) {
  test_resource tr;
  std::pmr::pool_options opts{0, 256};
  std::pmr::synchronized_pool_resource spr(opts, &tr);

  try {
    do_pattern(spr);
  } catch (const std::bad_alloc&) {
  }
  spr.release();

  assert(tr.successful_allocations.size() == tr.deallocations.size());
  assert(std::is_permutation(
      tr.successful_allocations.begin(),
      tr.successful_allocations.end(),
      tr.deallocations.begin(),
      tr.deallocations.end()));
}

template <std::size_t Bytes, size_t Align>
auto foo() {
  return [=](auto& mr) {
    void* p = mr.allocate(Bytes, Align);
    mr.deallocate(p, Bytes, Align);
  };
}

int main(int, char**) {
  test_allocation_pattern(foo<2, 1>());
  test_allocation_pattern(foo<2, 8>());
  test_allocation_pattern(foo<2, 64>());
  test_allocation_pattern(foo<128, 1>());
  test_allocation_pattern(foo<128, 8>());
  test_allocation_pattern(foo<128, 64>());
  test_allocation_pattern(foo<1024, 1>());
  test_allocation_pattern(foo<1024, 8>());
  test_allocation_pattern(foo<1024, 64>());

  test_allocation_pattern([](auto& mr) {
    void* p1 = mr.allocate(2, 1);
    void* p2 = mr.allocate(2, 8);
    void* p3 = mr.allocate(2, 64);
    void* p4 = mr.allocate(128, 1);
    void* p5 = mr.allocate(128, 8);
    void* p6 = mr.allocate(128, 64);
    void* p7 = mr.allocate(1024, 1);
    void* p8 = mr.allocate(1024, 8);
    void* p9 = mr.allocate(1024, 64);
    mr.deallocate(p1, 2, 1);
    mr.deallocate(p2, 2, 8);
    mr.deallocate(p3, 2, 64);
    mr.deallocate(p4, 128, 1);
    mr.deallocate(p5, 128, 8);
    mr.deallocate(p6, 128, 64);
    mr.deallocate(p7, 1024, 1);
    mr.deallocate(p8, 1024, 8);
    mr.deallocate(p9, 1024, 64);
  });

  return 0;
}
