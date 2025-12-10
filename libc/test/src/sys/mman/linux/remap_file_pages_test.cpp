//===-- Unittests for remap_file_pages ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/mman/remap_file_pages.h"
#include "src/unistd/close.h"
#include "src/unistd/sysconf.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/mman.h>
#include <sys/stat.h> // For S_IRWXU

const size_t PAGE_SIZE = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcRemapFilePagesTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcRemapFilePagesTest, NoError) {
  size_t page_size = PAGE_SIZE;
  ASSERT_GT(page_size, size_t(0));

  // Create a file-backed mapping
  constexpr const char *file_name = "remap_file_pages.test.noerror";
  auto test_file = libc_make_test_file_path(file_name);
  int fd = LIBC_NAMESPACE::open(test_file, O_RDWR | O_CREAT, S_IRWXU);
  ASSERT_GT(fd, 0);

  // First, allocate some memory using mmap
  size_t alloc_size = 2 * page_size;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_NE(addr, MAP_FAILED);

  // Now try to remap the pages
  EXPECT_THAT(LIBC_NAMESPACE::remap_file_pages(addr, page_size, 0, 1, 0),
              Succeeds());

  // Clean up
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, alloc_size), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds());
}

TEST_F(LlvmLibcRemapFilePagesTest, ErrorInvalidFlags) {
  size_t page_size = PAGE_SIZE;
  ASSERT_GT(page_size, size_t(0));

  // Create a file-backed mapping
  constexpr const char *file_name = "remap_file_pages.test.error";
  auto test_file = libc_make_test_file_path(file_name);
  int fd = LIBC_NAMESPACE::open(test_file, O_RDWR | O_CREAT, S_IRWXU);
  ASSERT_GT(fd, 0);

  // First, allocate some memory using mmap
  size_t alloc_size = 2 * page_size;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_NE(addr, MAP_FAILED);

  // Try to remap pages with an invalid flag MAP_PRIVATE
  EXPECT_THAT(LIBC_NAMESPACE::remap_file_pages(addr, page_size, PROT_READ, 0,
                                               MAP_PRIVATE),
              Fails(EINVAL));

  // Clean up
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::close(fd), Succeeds());
}

TEST_F(LlvmLibcRemapFilePagesTest, ErrorInvalidAddress) {
  size_t page_size = PAGE_SIZE;
  ASSERT_GT(page_size, size_t(0));

  // Use an address that we haven't mapped
  void *invalid_addr = reinterpret_cast<void *>(0x12345000);

  EXPECT_THAT(LIBC_NAMESPACE::remap_file_pages(invalid_addr, page_size,
                                               PROT_READ, 0, 0),
              Fails(EINVAL));
}
