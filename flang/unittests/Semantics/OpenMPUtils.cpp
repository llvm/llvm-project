//===- unittests/Semantics/OpenMPUtils.cpp  OpenMP utilities tests --------===//
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
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Semantics/openmp-utils.h"
#include "flang/Semantics/semantics.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "gtest/gtest.h"

using namespace Fortran;
using namespace Fortran::frontend;

namespace {

// This is a copy of FrontendActionTest.

class OpenMPUtilsTest : public ::testing::Test {
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
    invoc->getLangOpts().OpenMPVersion = 60;
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();

    compInst.setInvocation(std::move(invoc));
    compInst.getFrontendOpts().inputs.push_back(
        FrontendInputFile(inputFilePath, Language::Fortran));
    compInst.getFrontendOpts().features.Enable(common::LanguageFeature::OpenMP);
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

TEST_F(OpenMPUtilsTest, AffectedNestDepthNoClauses) {
  // Populate the input file with the pre-defined input and flush it.
  *inputFileOs << R"(
      integer :: i
      !$omp do
      do i = 1, 10
      end do
      end
  )";
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

  std::optional<parser::Program> &parseTree{compInst.getParsing().parseTree()};
  EXPECT_TRUE(parseTree.has_value());

  if (parseTree) {
    // clang-format off
    // The AST for the test program is
    // Program -> ProgramUnit -> MainProgram
    // | SpecificationPart
    // | | ImplicitPart ->
    // | ExecutionPart -> Block
    // | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPLoopConstruct
    // | | | OmpBeginLoopDirective
    // | | | | OmpDirectiveName -> llvm::omp::Directive = do
    // | | | | OmpClauseList ->
    // | | | | Flags = {}
    // | | | Block
    // | | | | ExecutionPartConstruct -> ExecutableConstruct -> DoConstruct
    // | | | | | NonLabelDoStmt
    // | | | | | | LoopControl -> LoopBounds
    // | | | | | | | Scalar -> Name = 'i'
    // | | | | | | | Scalar -> Expr -> LiteralConstant -> IntLiteralConstant = '1'
    // | | | | | | | Scalar -> Expr -> LiteralConstant -> IntLiteralConstant = '10'
    // | | | | | Block
    // | | | | | EndDoStmt ->
    // | EndProgramStmt ->
    // clang-format on
    auto &mainProgram =
        parser::UnwrapRef<parser::MainProgram>(parseTree->v.front());
    auto &body = std::get<parser::ExecutionPart>(mainProgram.t).v;
    auto &omp = parser::UnwrapRef<parser::OpenMPLoopConstruct>(body.front());
    auto [depth, mustBePerfect] =
        semantics::omp::GetAffectedNestDepthWithReason(omp.BeginDir(), 60);
    EXPECT_TRUE(depth.value.has_value());
    if (depth) {
      EXPECT_EQ(*depth.value, 1);
    }
  }
  // Validate the expected output.
  EXPECT_TRUE(success);
}

TEST_F(OpenMPUtilsTest, AffectedNestDepthCollapse) {
  // Populate the input file with the pre-defined input and flush it.
  *inputFileOs << R"(
      integer :: i, j
      !$omp do collapse(2)
      do i = 1, 10
        do j = 1, 10
        end do
      end do
      end
  )";
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

  std::optional<parser::Program> &parseTree{compInst.getParsing().parseTree()};
  EXPECT_TRUE(parseTree.has_value());

  if (parseTree) {
    auto &mainProgram =
        parser::UnwrapRef<parser::MainProgram>(parseTree->v.front());
    auto &body = std::get<parser::ExecutionPart>(mainProgram.t).v;
    auto &omp = parser::UnwrapRef<parser::OpenMPLoopConstruct>(body.front());
    auto [depth, mustBePerfect] =
        semantics::omp::GetAffectedNestDepthWithReason(omp.BeginDir(), 60);
    EXPECT_TRUE(depth.value.has_value());
    if (depth) {
      EXPECT_EQ(*depth.value, 2);
    }
  }
  // Validate the expected output.
  EXPECT_TRUE(success);
}

TEST_F(OpenMPUtilsTest, AffectedNestDepthCollapseOrdered) {
  // Populate the input file with the pre-defined input and flush it.
  *inputFileOs << R"(
      integer :: i, j, k, m
      !$omp do collapse(2) ordered(3)
      do i = 1, 10
        do j = 1, 10
          do k = 1, 10
            do m = 1, 10
            end do
          end do
        end do
      end do
      end
  )";
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

  std::optional<parser::Program> &parseTree{compInst.getParsing().parseTree()};
  EXPECT_TRUE(parseTree.has_value());

  if (parseTree) {
    auto &mainProgram =
        parser::UnwrapRef<parser::MainProgram>(parseTree->v.front());
    auto &body = std::get<parser::ExecutionPart>(mainProgram.t).v;
    auto &omp = parser::UnwrapRef<parser::OpenMPLoopConstruct>(body.front());
    auto [depth, mustBePerfect] =
        semantics::omp::GetAffectedNestDepthWithReason(omp.BeginDir(), 60);
    EXPECT_TRUE(depth.value.has_value());
    if (depth) {
      EXPECT_EQ(*depth.value, 3);
    }
  }
  // Validate the expected output.
  EXPECT_TRUE(success);
}

} // namespace
