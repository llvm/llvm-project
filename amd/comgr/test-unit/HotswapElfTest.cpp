//===- HotswapElfTest.cpp - Unit tests for HotSwap ELF layer --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-hotswap-internal.h"
#include "gtest/gtest.h"
#include <cstring>

using namespace COMGR::hotswap;

// -- ElfView::create ----------------------------------------------------------

TEST(ElfView, RejectsTruncatedInput) {
  uint8_t Garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
  llvm::Expected<ElfView> ViewOrErr = ElfView::create(Garbage, sizeof(Garbage));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}

TEST(ElfView, RejectsNonElfInput) {
  uint8_t NotElf[64] = {};
  llvm::Expected<ElfView> ViewOrErr = ElfView::create(NotElf, sizeof(NotElf));
  EXPECT_FALSE((bool)ViewOrErr);
  llvm::consumeError(ViewOrErr.takeError());
}
