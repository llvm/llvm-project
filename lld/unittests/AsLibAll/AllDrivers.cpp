//===- AllDrivers.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This test shows a typical case where all LLD drivers are linked into the
// application binary. This is very similar to how lld.exe binary is linked,
// except that here we cleanup the internal LLD memory context after each call.
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"

LLD_HAS_DRIVER(coff)
LLD_HAS_DRIVER(elf)
LLD_HAS_DRIVER(mingw)
LLD_HAS_DRIVER(macho)
LLD_HAS_DRIVER(wasm)

static bool lldInvoke(std::vector<const char *> args) {
  args.push_back("--version");
  lld::Result r =
      lld::lldMain(args, llvm::outs(), llvm::errs(), LLD_ALL_DRIVERS);
  return !r.retCode && r.canRunAgain;
}

TEST(AsLib, AllDrivers) {
  EXPECT_TRUE(lldInvoke({"ld.lld"}));
  EXPECT_TRUE(lldInvoke({"ld64.lld"}));
  EXPECT_TRUE(lldInvoke({"ld", "-m", "i386pe"})); // MinGW
  EXPECT_TRUE(lldInvoke({"lld-link"}));
  EXPECT_TRUE(lldInvoke({"wasm-ld"}));
}

TEST(AsLib, VirtualFileSystemRequiresELF) {
  auto fs = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  for (const char *name : {"ld64.lld", "lld-link", "wasm-ld", "ld.lld"}) {
    std::vector<const char *> args = {name, "--version"};
    if (llvm::StringRef(name) == "ld.lld")
      args.insert(args.end(), {"-m", "i386pep"});
    std::string diagnostics;
    llvm::raw_string_ostream err(diagnostics);
    lld::Result r = lld::lldMain(args, llvm::nulls(), err, LLD_ALL_DRIVERS, fs);
    EXPECT_NE(r.retCode, 0);
    EXPECT_TRUE(r.canRunAgain);
    EXPECT_THAT(diagnostics,
                testing::HasSubstr("only supported by the ELF driver"));
  }
}
