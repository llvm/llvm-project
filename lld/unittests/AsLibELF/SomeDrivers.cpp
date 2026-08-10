//===- SomeDrivers.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// In this test we showcase the fact that only one LLD driver can be invoked -
// the ELF driver that was linked in the test binary. Calling other drivers
// would return a failure. When using LLD as a library, any driver can be
// linked into your application.
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "gmock/gmock.h"

LLD_HAS_DRIVER(elf)

static bool lldInvoke(const char *lldExe) {
  std::vector<const char *> args{lldExe, "--version"};
  lld::Result s = lld::lldMain(args, llvm::outs(), llvm::errs(),
                               {{lld::Gnu, &lld::elf::link}});
  return !s.retCode && s.canRunAgain;
}

TEST(AsLib, SomeDrivers) {
  // When this flag is on, we actually need the MinGW driver library, not the
  // ELF one. Here our test only covers the case where the ELF driver is linked
  // into the unit test binary.
#ifndef LLD_DEFAULT_LD_LLD_IS_MINGW
  EXPECT_TRUE(lldInvoke("ld.lld")); // ELF
#endif
  // These drivers are not linked in this unit test.
  EXPECT_FALSE(lldInvoke("ld64.lld")); // Mach-O
  EXPECT_FALSE(lldInvoke("lld-link")); // COFF
  EXPECT_FALSE(lldInvoke("wasm-ld"));  // Wasm
}
