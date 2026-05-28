//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for flat_tlsf tag utilities.
///
//===----------------------------------------------------------------------===//

#include "src/__support/flat_tlsf/common.h"
#include "src/__support/flat_tlsf/tag.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::flat_tlsf::Byte;
namespace tag = LIBC_NAMESPACE::flat_tlsf::tag;

TEST(LlvmLibcFlatTlsfTagTest, ReadsFlags) {
  EXPECT_FALSE(tag::is_above_free(0));
  EXPECT_FALSE(tag::is_allocated(0));
  EXPECT_FALSE(tag::is_heap_base(0));
  EXPECT_FALSE(tag::is_heap_end(0));

  EXPECT_TRUE(tag::is_above_free(tag::ABOVE_FREE_FLAG));
  EXPECT_TRUE(tag::is_allocated(tag::ALLOCATED_FLAG));
  EXPECT_TRUE(tag::is_heap_base(tag::HEAP_BASE_FLAG));
  EXPECT_TRUE(tag::is_heap_end(tag::HEAP_END_FLAG));

  Byte all_flags = tag::ABOVE_FREE_FLAG | tag::ALLOCATED_FLAG |
                   tag::HEAP_BASE_FLAG | tag::HEAP_END_FLAG;
  EXPECT_TRUE(tag::is_above_free(all_flags));
  EXPECT_TRUE(tag::is_allocated(all_flags));
  EXPECT_TRUE(tag::is_heap_base(all_flags));
  EXPECT_TRUE(tag::is_heap_end(all_flags));
}

TEST(LlvmLibcFlatTlsfTagTest, MutatesFlags) {
  Byte byte = tag::ALLOCATED_FLAG;

  tag::set_above_free(&byte);
  EXPECT_TRUE(tag::is_above_free(byte));
  EXPECT_TRUE(tag::is_allocated(byte));

  // Test idempotency: setting already set flag should keep it set.
  tag::set_above_free(&byte);
  EXPECT_TRUE(tag::is_above_free(byte));
  EXPECT_TRUE(tag::is_allocated(byte));

  tag::clear_above_free(&byte);
  EXPECT_FALSE(tag::is_above_free(byte));
  EXPECT_TRUE(tag::is_allocated(byte));

  // Test idempotency: clearing already cleared flag should keep it clear.
  tag::clear_above_free(&byte);
  EXPECT_FALSE(tag::is_above_free(byte));
  EXPECT_TRUE(tag::is_allocated(byte));

  tag::set_end_flag(&byte);
  EXPECT_TRUE(tag::is_heap_end(byte));
  EXPECT_TRUE(tag::is_allocated(byte));

  // Test idempotency: setting already set flag should keep it set.
  tag::set_end_flag(&byte);
  EXPECT_TRUE(tag::is_heap_end(byte));
  EXPECT_TRUE(tag::is_allocated(byte));

  tag::clear_end_flag(&byte);
  EXPECT_FALSE(tag::is_heap_end(byte));
  EXPECT_TRUE(tag::is_allocated(byte));

  // Test idempotency: clearing already cleared flag should keep it clear.
  tag::clear_end_flag(&byte);
  EXPECT_FALSE(tag::is_heap_end(byte));
  EXPECT_TRUE(tag::is_allocated(byte));
}
