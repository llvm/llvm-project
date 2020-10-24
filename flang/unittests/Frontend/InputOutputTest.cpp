//===- unittests/Frontend/OutputStreamTest.cpp --- FrontendAction tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/FrontendTool/Utils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran::frontend;

namespace {

TEST(FrontendAction, TestInputOutputTestAction) {
  std::string inputFile = "io-file-test.f";
  std::error_code ec;

  // 1. Create the input file for the file manager
  // AllSources (which is used to manage files inside every compiler instance),
  // works with paths. This means that it requires a physical file. Create one.
  std::unique_ptr<llvm::raw_fd_ostream> os{
      new llvm::raw_fd_ostream(inputFile, ec, llvm::sys::fs::OF_None)};
  if (ec)
    FAIL() << "Failed to create the input file";

  // Populate the input file with the pre-defined input and flush it.
  *(os) << "End Program arithmetic";
  os.reset();

  // Get the path of the input file
  llvm::SmallString<64> cwd;
  if (std::error_code ec = llvm::sys::fs::current_path(cwd))
    FAIL() << "Failed to obtain the current working directory";
  std::string testFilePath(cwd.c_str());
  testFilePath += "/" + inputFile;

  // 2. Prepare the compiler (CompilerInvocation + CompilerInstance)
  CompilerInstance compInst;
  compInst.CreateDiagnostics();
  auto invocation = std::make_shared<CompilerInvocation>();
  invocation->GetFrontendOpts().programAction_ = InputOutputTest;
  compInst.SetInvocation(std::move(invocation));
  compInst.GetFrontendOpts().inputs_.push_back(
      FrontendInputFile(/*File=*/testFilePath, Language::Fortran));

  // 3. Set-up the output stream. Using output buffer wrapped as an output
  // stream, as opposed to an actual file (or a file descriptor).
  llvm::SmallVector<char, 256> outputFileBuffer;
  std::unique_ptr<llvm::raw_pwrite_stream> outputFileStream(
      new llvm::raw_svector_ostream(outputFileBuffer));
  compInst.SetOutputStream(std::move(outputFileStream));

  // 4. Run the earlier defined FrontendAction
  bool success = ExecuteCompilerInvocation(&compInst);

  EXPECT_TRUE(success);
  EXPECT_TRUE(!outputFileBuffer.empty());
  EXPECT_TRUE(llvm::StringRef(outputFileBuffer.data())
                  .startswith("End Program arithmetic"));

  // 5. Clear the input and the output files. Since we used an output buffer,
  // there are no physical output files to delete.
  ec = llvm::sys::fs::remove(inputFile);
  if (ec)
    FAIL() << "Failed to delete the test file";

  compInst.ClearOutputFiles(/*EraseFiles=*/false);
}
} // namespace
