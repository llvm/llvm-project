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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <memory>
#include <optional>
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

  std::optional<Driver> DriverInstance;
  std::unique_ptr<driver::Compilation> CompilationJob;
};

TEST_F(SanitizerArgsTest, Ignorelists) {
  const std::string ResourceDir = "/opt/llvm/lib/resources";
  const std::string UserIgnorelist = "/source/my_ignorelist.txt";
  const std::string ASanIgnorelist =
      concatPaths({ResourceDir, "share", "asan_ignorelist.txt"});

  auto &Command = emulateSingleCompilation(
      /*ExtraArgs=*/{"-fsanitize=address", "-resource-dir", ResourceDir,
                     std::string("-fsanitize-ignorelist=") + UserIgnorelist},
      /*ExtraFiles=*/{ASanIgnorelist, UserIgnorelist});

  // System ignorelists are added based on resource-dir.
  EXPECT_THAT(Command.getArguments(),
              Contains(StrEq(std::string("-fsanitize-system-ignorelist=") +
                             ASanIgnorelist)));
  // User ignorelists should also be added.
  EXPECT_THAT(
      Command.getArguments(),
      Contains(StrEq(std::string("-fsanitize-ignorelist=") + UserIgnorelist)));
}

TEST_F(SanitizerArgsTest, XRayLists) {
  const std::string XRayAllowlist = "/source/xray_allowlist.txt";
  const std::string XRayIgnorelist = "/source/xray_ignorelist.txt";
  const std::string XRayAttrList = "/source/xray_attr_list.txt";

  auto &Command = emulateSingleCompilation(
      /*ExtraArgs=*/
      {
          "-fxray-instrument",
          "-fxray-always-instrument=" + XRayAllowlist,
          "-fxray-never-instrument=" + XRayIgnorelist,
          "-fxray-attr-list=" + XRayAttrList,
      },
      /*ExtraFiles=*/{XRayAllowlist, XRayIgnorelist, XRayAttrList});

  // Ignorelists exist in the filesystem, so they should be added to the
  // compilation command, produced by the driver.
  EXPECT_THAT(Command.getArguments(),
              Contains(StrEq("-fxray-always-instrument=" + XRayAllowlist)));
  EXPECT_THAT(Command.getArguments(),
              Contains(StrEq("-fxray-never-instrument=" + XRayIgnorelist)));
  EXPECT_THAT(Command.getArguments(),
              Contains(StrEq("-fxray-attr-list=" + XRayAttrList)));
}

} // namespace
