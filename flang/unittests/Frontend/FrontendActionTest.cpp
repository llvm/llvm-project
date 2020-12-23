//===- unittests/Frontend/PrintPreprocessedTest.cpp  FrontendAction tests--===//
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

class FrontendActionTest : public ::testing::Test {
protected:
  // AllSources (which is used to manage files inside every compiler
  // instance), works with paths. So we need a filename and a path for the
  // input file.
  // TODO: We could use `-` for inputFilePath_, but then we'd need a way to
  // write to stdin that's then read by AllSources. Ideally, AllSources should
  // be capable of reading from any stream.
  std::string inputFileName_;
  std::string inputFilePath_;
  // The output stream for the input file. Use this to populate the input.
  std::unique_ptr<llvm::raw_fd_ostream> inputFileOs_;

  std::error_code ec_;

  CompilerInstance compInst_;
  std::shared_ptr<CompilerInvocation> invocation_;

  void SetUp() override {
    // Generate a unique test file name.
    const testing::TestInfo *const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    inputFileName_ = std::string(test_info->name()) + "_test-file.f";

    // Create the input file stream. Note that this stream is populated
    // separately in every test (i.e. the input is test specific).
    inputFileOs_ = std::make_unique<llvm::raw_fd_ostream>(
        inputFileName_, ec_, llvm::sys::fs::OF_None);
    if (ec_)
      FAIL() << "Failed to create the input file";

    // Get the path of the input file.
    llvm::SmallString<256> cwd;
    if (std::error_code ec_ = llvm::sys::fs::current_path(cwd))
      FAIL() << "Failed to obtain the current working directory";
    inputFilePath_ = cwd.c_str();
    inputFilePath_ += "/" + inputFileName_;

    // Prepare the compiler (CompilerInvocation + CompilerInstance)
    compInst_.CreateDiagnostics();
    invocation_ = std::make_shared<CompilerInvocation>();

    compInst_.set_invocation(std::move(invocation_));
    compInst_.frontendOpts().inputs_.push_back(
        FrontendInputFile(inputFilePath_, Language::Fortran));
  }

  void TearDown() override {
    // Clear the input file.
    llvm::sys::fs::remove(inputFileName_);

    // Clear the output files.
    // Note that these tests use an output buffer (as opposed to an output
    // file), hence there are no physical output files to delete and
    // `EraseFiles` is set to `false`. Also, some actions (e.g.
    // `ParseSyntaxOnly`) don't generated output. In such cases there's no
    // output to clear and `ClearOutputFile` returns immediately.
    compInst_.ClearOutputFiles(/*EraseFiles=*/false);
  }
};

TEST_F(FrontendActionTest, PrintPreprocessedInput) {
  // Populate the input file with the pre-defined input and flush it.
  *(inputFileOs_) << "#ifdef NEW\n"
                  << "  Program A \n"
                  << "#else\n"
                  << "  Program B\n"
                  << "#endif";
  inputFileOs_.reset();

  // Set-up the action kind.
  compInst_.invocation().frontendOpts().programAction_ = PrintPreprocessedInput;

  // Set-up the output stream. We are using output buffer wrapped as an output
  // stream, as opposed to an actual file (or a file descriptor).
  llvm::SmallVector<char, 256> outputFileBuffer;
  std::unique_ptr<llvm::raw_pwrite_stream> outputFileStream(
      new llvm::raw_svector_ostream(outputFileBuffer));
  compInst_.set_outputStream(std::move(outputFileStream));

  // Execute the action.
  bool success = ExecuteCompilerInvocation(&compInst_);

  // Validate the expected output.
  EXPECT_TRUE(success);
  EXPECT_TRUE(!outputFileBuffer.empty());
  EXPECT_TRUE(
      llvm::StringRef(outputFileBuffer.data()).startswith("program b\n"));
}

TEST_F(FrontendActionTest, ParseSyntaxOnly) {
  // Populate the input file with the pre-defined input and flush it.
  *(inputFileOs_) << "IF (A > 0.0) IF (B < 0.0) A = LOG (A)\n"
                  << "END";
  inputFileOs_.reset();

  // Set-up the action kind.
  compInst_.invocation().frontendOpts().programAction_ = ParseSyntaxOnly;

  // Set-up the output stream for the semantic diagnostics.
  llvm::SmallVector<char, 256> outputDiagBuffer;
  std::unique_ptr<llvm::raw_pwrite_stream> outputStream(
      new llvm::raw_svector_ostream(outputDiagBuffer));
  compInst_.set_semaOutputStream(std::move(outputStream));

  // Execute the action.
  bool success = ExecuteCompilerInvocation(&compInst_);

  // Validate the expected output.
  EXPECT_FALSE(success);
  EXPECT_TRUE(!outputDiagBuffer.empty());
  EXPECT_TRUE(
      llvm::StringRef(outputDiagBuffer.data())
          .startswith(
              ":1:14: error: IF statement is not allowed in IF statement\n"));
}
} // namespace
