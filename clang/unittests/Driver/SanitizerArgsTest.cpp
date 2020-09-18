//===- unittests/Driver/SanitizerArgsTest.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <memory>
#include <string>
using namespace clang;
using namespace clang::driver;

using ::testing::Contains;
using ::testing::StrEq;

namespace {

static constexpr const char *ClangBinary = "clang";
static constexpr const char *InputFile = "/sources/foo.c";

std::string concatPaths(llvm::ArrayRef<StringRef> Components) {
  llvm::SmallString<128> P;
  for (StringRef C : Components)
    llvm::sys::path::append(P, C);
  return std::string(P);
}

class SanitizerArgsTest : public ::testing::Test {
protected:
  const Command &emulateSingleCompilation(std::vector<std::string> ExtraArgs,
                                          std::vector<std::string> ExtraFiles) {
    assert(!DriverInstance && "Running twice is not allowed");

    llvm::IntrusiveRefCntPtr<DiagnosticOptions> Opts = new DiagnosticOptions;
    DiagnosticsEngine Diags(
        new DiagnosticIDs, Opts,
        new TextDiagnosticPrinter(llvm::errs(), Opts.get()));
    DriverInstance.emplace(ClangBinary, "x86_64-unknown-linux-gnu", Diags,
                           "clang LLVM compiler", prepareFS(ExtraFiles));

    std::vector<const char *> Args = {ClangBinary};
    for (const auto &A : ExtraArgs)
      Args.push_back(A.c_str());
    Args.push_back("-c");
    Args.push_back(InputFile);

    CompilationJob.reset(DriverInstance->BuildCompilation(Args));

    if (Diags.hasErrorOccurred())
      ADD_FAILURE() << "Error occurred while parsing compilation arguments. "
                       "See stderr for details.";

    const auto &Commands = CompilationJob->getJobs().getJobs();
    assert(Commands.size() == 1);
    return *Commands.front();
  }

private:
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>
  prepareFS(llvm::ArrayRef<std::string> ExtraFiles) {
    llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS =
        new llvm::vfs::InMemoryFileSystem;
    FS->addFile(ClangBinary, time_t(), llvm::MemoryBuffer::getMemBuffer(""));
    FS->addFile(InputFile, time_t(), llvm::MemoryBuffer::getMemBuffer(""));
    for (llvm::StringRef F : ExtraFiles)
      FS->addFile(F, time_t(), llvm::MemoryBuffer::getMemBuffer(""));
    return FS;
  }

  llvm::Optional<Driver> DriverInstance;
  std::unique_ptr<driver::Compilation> CompilationJob;
};

TEST_F(SanitizerArgsTest, Blacklists) {
  const std::string ResourceDir = "/opt/llvm/lib/resources";
  const std::string UserBlacklist = "/source/my_blacklist.txt";
  const std::string ASanBlacklist =
      concatPaths({ResourceDir, "share", "asan_blacklist.txt"});

  auto &Command = emulateSingleCompilation(
      /*ExtraArgs=*/{"-fsanitize=address", "-resource-dir", ResourceDir,
                     std::string("-fsanitize-blacklist=") + UserBlacklist},
      /*ExtraFiles=*/{ASanBlacklist, UserBlacklist});

  // System blacklists are added based on resource-dir.
  EXPECT_THAT(Command.getArguments(),
              Contains(StrEq(std::string("-fsanitize-system-blacklist=") +
                             ASanBlacklist)));
  // User blacklists should also be added.
  EXPECT_THAT(
      Command.getArguments(),
      Contains(StrEq(std::string("-fsanitize-blacklist=") + UserBlacklist)));
}

TEST_F(SanitizerArgsTest, XRayLists) {
  const std::string XRayWhitelist = "/source/xray_whitelist.txt";
  const std::string XRayBlacklist = "/source/xray_blacklist.txt";
  const std::string XRayAttrList = "/source/xray_attr_list.txt";

  auto &Command = emulateSingleCompilation(
      /*ExtraArgs=*/
      {
          "-fxray-instrument",
          "-fxray-always-instrument=" + XRayWhitelist,
          "-fxray-never-instrument=" + XRayBlacklist,
          "-fxray-attr-list=" + XRayAttrList,
      },
      /*ExtraFiles=*/{XRayWhitelist, XRayBlacklist, XRayAttrList});

  // Blacklists exist in the filesystem, so they should be added to the
  // compilation command, produced by the driver.
  EXPECT_THAT(Command.getArguments(),
              Contains(StrEq("-fxray-always-instrument=" + XRayWhitelist)));
  EXPECT_THAT(Command.getArguments(),
              Contains(StrEq("-fxray-never-instrument=" + XRayBlacklist)));
  EXPECT_THAT(Command.getArguments(),
              Contains(StrEq("-fxray-attr-list=" + XRayAttrList)));
}

} // namespace
