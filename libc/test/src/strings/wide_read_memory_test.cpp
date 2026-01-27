//===-- Memory bounds check test for wide-read functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For performance, some vector-based libc functions read data outside of, but
// adjacent to, the input address. For example, string_length can read both
// before and after the data in its src parameter. As part of the
// implementation, it is allowed to do this. However, the code must take care to
// avoid address errors. The sanitizers can't distinguish between "the
// implementation" and user-code, and so report an error. Therefore we can't use
// them to check if functions like these have memory errors.
//
// This test uses mprotect to simulate address sanitization. Tests that read too
// far outside data will segfault.
//
// It creates three adjacent pages in memory. The outer two are mprotected
// unreadable, the middle usable normally. By placing test data at the edges
// between the middle page and the others, we can test for bad accesses.

#include "src/__support/CPP/array.h"
#include "src/string/memory_utils/inline_memset.h"
#include "src/string/string_utils.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/mprotect.h"
#include "src/sys/mman/munmap.h"
#include "src/unistd/getpagesize.h"
#include "test/UnitTest/MemoryMatcher.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

using TwoKilobyteBuffer = cpp::array<char, 2048>;
// This could be smaller on a target-basis, but that adds complexity and the
// extra testing is fine.
static constexpr unsigned long kLargestTestVectorSize = 512;

class LlvmLibcWideAccessMemoryTest : public testing::Test {
  char *page0_;
  char *page1_;
  char *page2_;
  size_t page_size;

public:
  void SetUp() override {
    page_size = LIBC_NAMESPACE::getpagesize();
    page0_ = static_cast<char *>(
        LIBC_NAMESPACE::mmap(nullptr, page_size * 3, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(static_cast<void *>(page0_), MAP_FAILED);
    page1_ = page0_ + page_size;
    page2_ = page1_ + page_size;
    LIBC_NAMESPACE::mprotect(page0_, page_size, PROT_NONE);
    LIBC_NAMESPACE::mprotect(page2_, page_size, PROT_NONE);
  }

  void TearDown() override { LIBC_NAMESPACE::munmap(page0_, page_size * 3); }

  // Repeatedly runs "func" on copies of the data in "buf", each progressively
  // closer to the boundary of valid memory. Test will segfault if function
  // under test accesses invalid memory.
  //
  // Func should test the function in question just as normal. Recommend making
  // the amount of test data at least 1.5k, which guarantees a wind-up, multiple
  // iterations of the inner loop, and a wind-down, even on systems with
  // 512-byte vectors. The termination condition, eg, end-of string or character
  // being searched for, should be near the end of the data.
  template <typename TestFunc>
  void TestMemoryAccess(const TwoKilobyteBuffer &buf, TestFunc func) {
    // Run func on data near the start boundary of valid memory.
    for (unsigned long offset = 0; offset < kLargestTestVectorSize; ++offset) {
      char *test_addr = page1_ + offset;
      inline_memcpy(test_addr, buf.data(), buf.size());
      func(test_addr);
    }
    // Run func on data near the end boundary of valid memory.
    for (unsigned long offset = 0; offset < kLargestTestVectorSize; ++offset) {
      char *test_addr = page2_ - buf.size() - offset - 1;
      ASSERT_LE(test_addr + buf.size(), page2_);
      inline_memcpy(test_addr, buf.data(), buf.size());
      func(test_addr);
    }
  }
};

TEST_F(LlvmLibcWideAccessMemoryTest, StringLength) {
  // 1.5 k long vector of a's.
  TwoKilobyteBuffer buf;
  inline_memset(buf.data(), 'a', buf.size());
  // Make sure it is null terminated.
  buf[buf.size() - 1] = '\0';
  this->TestMemoryAccess(buf, [this, buf](const char *test_data) {
    // -1 for the null character.
    ASSERT_EQ(internal::string_length(test_data), size_t(buf.size() - 1));
  });
}

} // namespace LIBC_NAMESPACE_DECL
