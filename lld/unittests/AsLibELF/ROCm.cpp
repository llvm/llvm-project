//===- ROCm.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The purpose of this test is to showcase a more singular usage of LLD as a
// library, where only one LLD driver is being used (and linked in the target
// application). We also expect that linking twice the same object files
// would yield a successfull result. When used as library, LLD always cleans its
// internal memory context after each linker call.
//===----------------------------------------------------------------------===//

// When this flag is on, we actually need the MinGW driver library, not the
// ELF one. Here our test only covers the case where the ELF driver is linked
// into the unit test binary.
#ifndef LLD_DEFAULT_LD_LLD_IS_MINGW

#include "lld/Common/Driver.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"

extern const char *TestMainArgv0;

LLD_HAS_DRIVER(elf)

static bool lldInvoke(const char *inPath, const char *outPath) {
  std::vector<const char *> args{"ld.lld", "-shared", inPath, "-o", outPath};
  lld::Result s = lld::lldMain(args, llvm::outs(), llvm::errs(),
                               {{lld::Gnu, &lld::elf::link}});
  return !s.retCode && s.canRunAgain;
}

static bool runLinker(llvm::StringRef input) {
  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  llvm::SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                         tempHsacoFilename)) {
    return false;
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);
  // Invoke lld. Expect a true return value from lld.
  llvm::SmallString<128> inputPath =
      llvm::unittest::getInputFileDirectory(TestMainArgv0);
  llvm::sys::path::append(inputPath, input);
  if (!lldInvoke(inputPath.c_str(), tempHsacoFilename.c_str())) {
    llvm::errs() << "Failed to link: " << inputPath << "\n";
    return false;
  }
  return true;
}

TEST(AsLib, ROCm) {
  EXPECT_TRUE(runLinker("kernel1.o"));
  EXPECT_TRUE(runLinker("kernel2.o"));
  EXPECT_TRUE(runLinker("kernel1.o"));
  EXPECT_TRUE(runLinker("kernel2.o"));
}
#endif
