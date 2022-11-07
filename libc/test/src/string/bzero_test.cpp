//===-- Unittests for bzero -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory_utils/memory_check_utils.h"
#include "src/string/bzero.h"
#include "utils/UnitTest/Test.h"

namespace __llvm_libc {

// Adapt CheckMemset signature to op implementation signatures.
template <auto FnImpl>
void BzeroAdaptor(cpp::span<char> p1, uint8_t value, size_t size) {
  assert(value == 0);
  FnImpl(p1.begin(), size);
}

TEST(LlvmLibcBzeroTest, SizeSweep) {
  static constexpr size_t kMaxSize = 1024;
  static constexpr auto Impl = BzeroAdaptor<__llvm_libc::bzero>;
  Buffer DstBuffer(kMaxSize);
  for (size_t size = 0; size < kMaxSize; ++size) {
    auto dst = DstBuffer.span().subspan(0, size);
    ASSERT_TRUE((CheckMemset<Impl>(dst, 0, size)));
  }
}

} // namespace __llvm_libc
