//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_DEFAULT_LD_LLD_IS_MINGW

#include "lld/Common/Driver.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"

LLD_HAS_DRIVER(elf)

// With "-o -", the ELF driver writes the output image to the stdoutOS stream
// passed to link(), so lld used as a library can capture it in a raw_ostream.
TEST(AsLib, OutputToStream) {
  llvm::SmallString<256> input(getenv("LLD_SRC_DIR"));
  llvm::sys::path::append(input, "unittests", "AsLibELF", "Inputs",
                          "kernel1.o");
  std::vector<const char *> args{"ld.lld", "-shared", input.c_str(), "-o", "-"};

  std::string buf;
  llvm::raw_string_ostream os(buf);
  lld::Result s =
      lld::lldMain(args, os, llvm::errs(), {{lld::Gnu, &lld::elf::link}});
  EXPECT_EQ(s.retCode, 0);
  EXPECT_TRUE(s.canRunAgain);
  EXPECT_TRUE(llvm::StringRef(buf).starts_with("\177ELF"));
}
#endif
