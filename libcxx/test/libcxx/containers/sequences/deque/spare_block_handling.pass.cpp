//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <deque>

// Test how deque manages the spare blocks it keeps. The exact values it tests
// for are not always important, but are sometimes needed to ensure the container
// resizes or shrinks at the correct time.

#include <deque>
#include <cassert>
#include <cstdio>
#include <memory>
#include <queue>
#include <stack>
#include <string>

#include "min_allocator.h"
#include "assert_macros.h"

template <class Adaptor>
struct ContainerAdaptor : public Adaptor {
  using Adaptor::Adaptor;
  typename Adaptor::container_type& GetContainer() { return Adaptor::c; }
};

template <class Deque>
static void print(const Deque& d) {
  std::printf("%zu : __front_spare() == %zu"
              " : __back_spare() == %zu"
              " : __capacity() == %zu"
              " : bytes allocated == %zu\n",
              d.size(), d.__front_spare(), d.__back_spare(), d.__capacity(),
              malloc_allocator_base::outstanding_bytes);
}

template <class T>
using Deque = std::deque<T, malloc_allocator<T> >;

template <class T>
using BlockSize = std::__deque_block_size<T, std::ptrdiff_t>;

struct LargeT {
  LargeT() = default;
  char buff[256] = {};
};
static_assert(BlockSize<LargeT>::value == 16, "");

const auto& AllocBytes = malloc_allocator_base::outstanding_bytes;

template <class Deque>
struct PrintOnFailure {
   explicit PrintOnFailure(Deque const& deque) : deque_(&deque) {}
   void operator()() const { print(*deque_); }
private:
  const Deque* deque_;

  PrintOnFailure(PrintOnFailure const&) = delete;
};


static void push_back() {
  const auto BS = BlockSize<LargeT>::value;
  std::unique_ptr<Deque<LargeT>> dp(new Deque<LargeT>);
  auto& d = *dp;
  PrintOnFailure<Deque<LargeT>> on_fail(d);

  // Test nothing is allocated after default construction.
  {
    TEST_REQUIRE(d.size() == 0, on_fail);
    TEST_REQUIRE(d.__capacity() == 0, on_fail);
    TEST_REQUIRE(d.__block_count() == 0, on_fail);
  }
  // First push back allocates one block.
  d.push_back({});
  {
    TEST_REQUIRE(d.size() == 1, on_fail);
    TEST_REQUIRE(d.__front_spare() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare() == 14, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__capacity() == BS - 1, on_fail);
    TEST_REQUIRE(d.__block_count() == 1, on_fail);
  }

  d.push_back({});
  {
    TEST_REQUIRE(d.size() == 2, on_fail);
    TEST_REQUIRE(d.__front_spare() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare() == 13, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 0, on_fail);
  }
  // Push back until we need a new block.
  for (int RemainingCap = d.__capacity() - d.size(); RemainingCap >= 0; --RemainingCap)
    d.push_back({});
  {
    TEST_REQUIRE(d.__block_count() == 2, on_fail);
    TEST_REQUIRE(d.__front_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare() == 15, on_fail);
  }

  // Remove the only element in the new block. Test that we keep the empty
  // block as a spare.
  d.pop_back();
  {
    TEST_REQUIRE(d.__block_count() == 2, on_fail);
    TEST_REQUIRE(d.__front_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 1, on_fail);
    TEST_REQUIRE(d.__back_spare() == 16, on_fail);
  }

  // Pop back again, keep the spare.
  d.pop_back();
  {
    TEST_REQUIRE(d.__block_count() == 2, on_fail);
    TEST_REQUIRE(d.__front_spare() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare() == 17, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 1, on_fail);
  }

  dp.reset();
  TEST_REQUIRE(AllocBytes == 0, on_fail);
}

static void push_front() {
  std::unique_ptr<Deque<LargeT>> dp(new Deque<LargeT>);
  auto& d = *dp;
  PrintOnFailure<Deque<LargeT>> on_fail(d);

  // Test nothing is allocated after default construction.
  {
    TEST_REQUIRE(d.size() == 0, on_fail);
    TEST_REQUIRE(d.__capacity() == 0, on_fail);
    TEST_REQUIRE(d.__block_count() == 0, on_fail);
  }
  // First push front allocates one block, and we start the sequence in the
  // middle.
  d.push_front({});
  {
    TEST_REQUIRE(d.size() == 1, on_fail);
    TEST_REQUIRE(d.__front_spare() == 7, on_fail);
    TEST_REQUIRE(d.__back_spare() == 7, on_fail);
    TEST_REQUIRE(d.__front_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__block_count() == 1, on_fail);
  }

  d.push_front({});
  {
    TEST_REQUIRE(d.size() == 2, on_fail);
    TEST_REQUIRE(d.__front_spare() == 6, on_fail);
    TEST_REQUIRE(d.__back_spare() == 7, on_fail);
    TEST_REQUIRE(d.__front_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 0, on_fail);
  }
  // Push front until we need a new block.
  for (int RemainingCap = d.__front_spare(); RemainingCap >= 0; --RemainingCap)
    d.push_front({});
  {
    TEST_REQUIRE(d.__block_count() == 2, on_fail);
    TEST_REQUIRE(d.__front_spare() == 15, on_fail);
    TEST_REQUIRE(d.__back_spare() == 7, on_fail);
    TEST_REQUIRE(d.__front_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 0, on_fail);
  }

  // Remove the only element in the new block. Test that we keep the empty
  // block as a spare.
  d.pop_front();
  {
    TEST_REQUIRE(d.__block_count() == 2, on_fail);
    TEST_REQUIRE(d.__front_spare_blocks() == 1, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare() == 7, on_fail);
  }

  // Pop back again, keep the spare.
  d.pop_front();
  {
    TEST_REQUIRE(d.__block_count() == 2, on_fail);
    TEST_REQUIRE(d.__front_spare_blocks() == 1, on_fail);
    TEST_REQUIRE(d.__back_spare() == 7, on_fail);
  }

  dp.reset();
  TEST_REQUIRE(AllocBytes == 0, on_fail);
}

static void std_queue() {
  using D = Deque<LargeT>;
  using Queue = std::queue<LargeT, D>;
  ContainerAdaptor<Queue> CA;
  const D& d = CA.GetContainer();
  Queue &q = CA;
  PrintOnFailure<Deque<LargeT>> on_fail(d);

  while (d.__block_count() < 4)
    q.push({});
  {
    TEST_REQUIRE(d.__block_count() == 4, on_fail);
    TEST_REQUIRE(d.__front_spare() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare() == 15, on_fail);
    TEST_REQUIRE(d.__back_spare_blocks() == 0, on_fail);
  }
  while (d.__back_spare()) {
    q.push({});
  }
  {
    TEST_REQUIRE(d.__block_count() == 4, on_fail);
    TEST_REQUIRE(d.__front_spare() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare() == 0, on_fail);
  }
  q.pop();
  {
    TEST_REQUIRE(d.__block_count() == 4, on_fail);
    TEST_REQUIRE(d.__front_spare() == 1, on_fail);
    TEST_REQUIRE(d.__back_spare() == 0, on_fail);
  }

  // Pop until we create a spare block at the front.
  while (d.__front_spare() <= 15)
    q.pop();

  {
    TEST_REQUIRE(d.__block_count() == 4, on_fail);
    TEST_REQUIRE(d.__front_spare_blocks() == 1, on_fail);
    TEST_REQUIRE(d.__front_spare() == 16, on_fail);
    TEST_REQUIRE(d.__back_spare() == 0, on_fail);
  }

  // Push at the end -- should re-use new spare block at front
  q.push({});

  {
    TEST_REQUIRE(d.__block_count() == 4, on_fail);
    TEST_REQUIRE(d.__front_spare_blocks() == 0, on_fail);
    TEST_REQUIRE(d.__front_spare() == 0, on_fail);
    TEST_REQUIRE(d.__back_spare() == 15, on_fail);
  }
  while (!q.empty()) {
    q.pop();
    TEST_REQUIRE(d.__front_spare_blocks() + d.__back_spare_blocks() <= 2, on_fail);
  }

  // The empty state has two blocks
  {
    TEST_REQUIRE(d.__front_spare() == 16, on_fail);
    TEST_REQUIRE(d.__back_spare() == 15, on_fail);
    TEST_REQUIRE(d.__capacity() == 31, on_fail);
  }
}

static void pop_front_push_back() {
  Deque<char> d(32 * 1024, 'a');
  bool take_from_front = true;
  while (d.size() > 0) {
    if (take_from_front) {
      d.pop_front();
      take_from_front = false;
    } else {
      d.pop_back();
      take_from_front = true;
    }
    if (d.size() % 1000 == 0 || d.size() < 50) {
      print(d);
    }
  }
}

int main(int, char**) {
  push_back();
  push_front();
  std_queue();
  pop_front_push_back();

  return 0;
}
