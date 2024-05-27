//===- unittests/Frontend/FrontendActionTest.cpp  FrontendAction tests-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/FrontendTool/Utils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "gtest/gtest.h"

using namespace Fortran::frontend;

namespace {

class FrontendActionTest : public ::testing::Test {
protected:
  // AllSources (which is used to manage files inside every compiler
  // instance), works with paths. So we need a filename and a path for the
  // input file.
  // TODO: We could use `-` for inputFilePath, but then we'd need a way to
  // write to stdin that's then read by AllSources. Ideally, AllSources should
  // be capable of reading from any stream.
  std::string inputFileName;
  std::string inputFilePath;
  // The output stream for the input file. Use this to populate the input.
  std::unique_ptr<llvm::raw_fd_ostream> inputFileOs;

  std::error_code ec;

  CompilerInstance compInst;
  std::shared_ptr<CompilerInvocation> invoc;

  void SetUp() override {
    // Generate a unique test file name.
    const testing::TestInfo *const testInfo =
        testing::UnitTest::GetInstance()->current_test_info();
    inputFileName = std::string(testInfo->name()) + "_test-file.f90";

    // Create the input file stream. Note that this stream is populated
    // separately in every test (i.e. the input is test specific).
    inputFileOs = std::make_unique<llvm::raw_fd_ostream>(
        inputFileName, ec, llvm::sys::fs::OF_None);
    if (ec)
      FAIL() << "Failed to create the input file";

    // Get the path of the input file.
    llvm::SmallString<256> cwd;
    if (std::error_code ec = llvm::sys::fs::current_path(cwd))
      FAIL() << "Failed to obtain the current working directory";
    inputFilePath = cwd.c_str();
    inputFilePath += "/" + inputFileName;

    // Prepare the compiler (CompilerInvocation + CompilerInstance)
    compInst.createDiagnostics();
    invoc = std::make_shared<CompilerInvocation>();

    // Set-up default target triple and initialize LLVM Targets so that the
    // target data layout can be passed to the frontend.
    invoc->getTargetOpts().triple =
        llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();

    compInst.setInvocation(std::move(invoc));
    compInst.getFrontendOpts().inputs.push_back(
        FrontendInputFile(inputFilePath, Language::Fortran));
  }

  void TearDown() override {
    // Clear the input file.
    llvm::sys::fs::remove(inputFileName);

    // Clear the output files.
    // Note that these tests use an output buffer (as opposed to an output
    // file), hence there are no physical output files to delete and
    // `EraseFiles` is set to `false`. Also, some actions (e.g.
    // `ParseSyntaxOnly`) don't generated output. In such cases there's no
    // output to clear and `ClearOutputFile` returns immediately.
    compInst.clearOutputFiles(/*EraseFiles=*/false);
  }
};

TEST_F(FrontendActionTest, TestInputOutput) {
  // Populate the input file with the pre-defined input and flush it.
  *(inputFileOs) << "End Program arithmetic";
  inputFileOs.reset();

  // Set-up the action kind.
  compInst.getInvocation().getFrontendOpts().programAction = InputOutputTest;

  // Set-up the output stream. Using output buffer wrapped as an output
  // stream, as opposed to an actual file (or a file descriptor).
  llvm::SmallVector<char, 256> outputFileBuffer;
  std::unique_ptr<llvm::raw_pwrite_stream> outputFileStream(
      new llvm::raw_svector_ostream(outputFileBuffer));
  compInst.setOutputStream(std::move(outputFileStream));

  // Execute the action.
  bool success = executeCompilerInvocation(&compInst);

  // Validate the expected output.
  EXPECT_TRUE(success);
  EXPECT_TRUE(!outputFileBuffer.empty());
  EXPECT_TRUE(llvm::StringRef(outputFileBuffer.data())
                  .starts_with("End Program arithmetic"));
}

TEST_F(FrontendActionTest, PrintPreprocessedInput) {
  // Populate the input file with the pre-defined input and flush it.
  *(inputFileOs) << "#ifdef NEW\n"
                 << "  Program A \n"
                 << "#else\n"
                 << "  Program B\n"
                 << "#endif";
  inputFileOs.reset();

  // Set-up the action kind.
  compInst.getInvocation().getFrontendOpts().programAction =
      PrintPreprocessedInput;
  compInst.getInvocation().getPreprocessorOpts().noReformat = true;

  // Set-up the output stream. We are using output buffer wrapped as an output
  // stream, as opposed to an actual file (or a file descriptor).
  llvm::SmallVector<char, 256> outputFileBuffer;
  std::unique_ptr<llvm::raw_pwrite_stream> outputFileStream(
      new llvm::raw_svector_ostream(outputFileBuffer));
  compInst.setOutputStream(std::move(outputFileStream));

  // Execute the action.
  bool success = executeCompilerInvocation(&compInst);

  // Validate the expected output.
  EXPECT_TRUE(success);
  EXPECT_TRUE(!outputFileBuffer.empty());
  EXPECT_TRUE(
      llvm::StringRef(outputFileBuffer.data()).starts_with("program b\n"));
}

TEST_F(FrontendActionTest, ParseSyntaxOnly) {
  // Populate the input file with the pre-defined input and flush it.
  *(inputFileOs) << "IF (A > 0.0) IF (B < 0.0) A = LOG (A)\n"
                 << "END";
  inputFileOs.reset();

  // Set-up the action kind.
  compInst.getInvocation().getFrontendOpts().programAction = ParseSyntaxOnly;

  // Set-up the output stream for the semantic diagnostics.
  llvm::SmallVector<char, 256> outputDiagBuffer;
  std::unique_ptr<llvm::raw_pwrite_stream> outputStream(
      new llvm::raw_svector_ostream(outputDiagBuffer));
  compInst.setSemaOutputStream(std::move(outputStream));

  // Execute the action.
  bool success = executeCompilerInvocation(&compInst);

  // Validate the expected output.
  EXPECT_FALSE(success);
  EXPECT_TRUE(!outputDiagBuffer.empty());
  EXPECT_TRUE(
      llvm::StringRef(outputDiagBuffer.data())
          .contains(
              ":1:14: error: IF statement is not allowed in IF statement\n"));
}

TEST_F(FrontendActionTest, EmitLLVM) {
  // Populate the input file with the pre-defined input and flush it.
  *(inputFileOs) << "end program";
  inputFileOs.reset();

  // Set-up the action kind.
  compInst.getInvocation().getFrontendOpts().programAction = EmitLLVM;

  // Initialise LLVM backend
  llvm::InitializeAllAsmPrinters();

  // Set-up the output stream. We are using output buffer wrapped as an output
  // stream, as opposed to an actual file (or a file descriptor).
  llvm::SmallVector<char> outputFileBuffer;
  std::unique_ptr<llvm::raw_pwrite_stream> outputFileStream(
      new llvm::raw_svector_ostream(outputFileBuffer));
  compInst.setOutputStream(std::move(outputFileStream));

  // Execute the action.
  bool success = executeCompilerInvocation(&compInst);

  // Validate the expected output.
  EXPECT_TRUE(success);
  EXPECT_TRUE(!outputFileBuffer.empty());

  EXPECT_TRUE(llvm::StringRef(outputFileBuffer.begin(), outputFileBuffer.size())
                  .contains("define void @_QQmain()"));
}

TEST_F(FrontendActionTest, EmitAsm) {
  // Populate the input file with the pre-defined input and flush it.
  *(inputFileOs) << "end program";
  inputFileOs.reset();

  // Set-up the action kind.
  compInst.getInvocation().getFrontendOpts().programAction = EmitAssembly;

  // Initialise LLVM backend
  llvm::InitializeAllAsmPrinters();

  // Set-up the output stream. We are using output buffer wrapped as an output
  // stream, as opposed to an actual file (or a file descriptor).
  llvm::SmallVector<char, 256> outputFileBuffer;
  std::unique_ptr<llvm::raw_pwrite_stream> outputFileStream(
      new llvm::raw_svector_ostream(outputFileBuffer));
  compInst.setOutputStream(std::move(outputFileStream));

  // Execute the action.
  bool success = executeCompilerInvocation(&compInst);

  // Validate the expected output.
  EXPECT_TRUE(success);
  EXPECT_TRUE(!outputFileBuffer.empty());

  EXPECT_TRUE(llvm::StringRef(outputFileBuffer.begin(), outputFileBuffer.size())
                  .contains("_QQmain"));
}
} // namespace
