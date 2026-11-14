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
#include "flang/Lower/LoweringOptions.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/unparse.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "gtest/gtest.h"

using namespace Fortran::frontend;

namespace {

struct RecoveryNodeVisitor {
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}

  bool Pre(const Fortran::parser::ProgramUnit &x) {
    program += std::holds_alternative<Fortran::parser::ErrorRecovery>(x.u);
    return true;
  }
  bool Pre(const Fortran::parser::InternalSubprogram &x) {
    internal += std::holds_alternative<Fortran::parser::ErrorRecovery>(x.u);
    return true;
  }
  bool Pre(const Fortran::parser::ModuleSubprogram &x) {
    module += std::holds_alternative<Fortran::parser::ErrorRecovery>(x.u);
    return true;
  }
  bool Pre(const Fortran::parser::InterfaceSpecification &x) {
    interface += std::holds_alternative<Fortran::parser::ErrorRecovery>(x.u);
    return true;
  }

  unsigned program{0}, internal{0}, module{0}, interface{0};
};

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

  // Exercise every list whose variant now includes an ErrorRecovery node.
  void parseRecoveryNodes() {
    *(inputFileOs) << "!$acc declare create(x)\n"
                   << "subroutine outer()\n"
                   << "interface\n"
                   << "!$acc declare create(x)\n"
                   << "subroutine iface()\n"
                   << "end subroutine\n"
                   << "end interface\n"
                   << "contains\n"
                   << "!$acc declare create(x)\n"
                   << "subroutine inner()\n"
                   << "end subroutine\n"
                   << "end subroutine\n"
                   << "module m\n"
                   << "contains\n"
                   << "!$acc declare create(x)\n"
                   << "subroutine child()\n"
                   << "end subroutine\n"
                   << "end module\n";
    inputFileOs.reset();

    auto &options = compInst.getInvocation().getFortranOpts();
    options.features.Enable(Fortran::common::LanguageFeature::OpenACC);
    auto &parsing = compInst.getParsing();
    ASSERT_NE(parsing.Prescan(inputFilePath, options), nullptr);
    ASSERT_FALSE(parsing.messages().AnyFatalError());
    parsing.Parse(llvm::nulls(), compInst.getInvocation().getLangOpts());
    ASSERT_TRUE(parsing.consumedWholeFile());
    ASSERT_TRUE(parsing.messages().AnyFatalError());
    ASSERT_TRUE(parsing.parseTree().has_value());
  }

  // Targets register their library options, which compInst holds from its
  // construction.
  static void SetUpTestSuite() {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
  }

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

    // Set-up default target triple so that the target data layout can be
    // passed to the frontend.
    compInst.getInvocation().getTargetOpts().triple =
        llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());

    compInst.createDiagnostics();
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
      llvm::StringRef(outputFileBuffer.data()).starts_with(" program b\n"));
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

TEST_F(FrontendActionTest, RecoverOpenACCDeclarePreservesDirectives) {
  *(inputFileOs) << "!$acc declare create(x)\n"
                 << "!$acc declare create(y)\n"
                 << "!$acc routine(s) seq\n"
                 << "!dir$ ivdep\n"
                 << "!$acc declare create(z)\n"
                 << "!$acc declare create(w)\n"
                 << "!dir$ ivdep; subroutine s()\n"
                 << "end subroutine\n";
  inputFileOs.reset();

  auto &options = compInst.getInvocation().getFortranOpts();
  options.features.Enable(Fortran::common::LanguageFeature::OpenACC);
  options.compilerDirectiveSentinels.emplace_back("dir$");
  auto &parsing = compInst.getParsing();
  ASSERT_NE(parsing.Prescan(inputFilePath, options), nullptr);
  ASSERT_FALSE(parsing.messages().AnyFatalError());
  parsing.Parse(llvm::nulls(), compInst.getInvocation().getLangOpts());
  ASSERT_TRUE(parsing.consumedWholeFile());
  ASSERT_TRUE(parsing.messages().AnyFatalError());
  ASSERT_TRUE(parsing.parseTree().has_value());

  // Consecutive DECLAREs share a recovery node. Other directives and the
  // following procedure retain their positions and their actual node types.
  const auto &units = parsing.parseTree()->v;
  ASSERT_EQ(units.size(), 6u);
  auto unit = units.begin();
  EXPECT_TRUE(
      std::holds_alternative<Fortran::parser::ErrorRecovery>(unit++->u));
  EXPECT_TRUE(std::holds_alternative<
      Fortran::common::Indirection<Fortran::parser::OpenACCRoutineConstruct>>(
      unit++->u));
  EXPECT_TRUE(std::holds_alternative<
      Fortran::common::Indirection<Fortran::parser::CompilerDirective>>(
      unit++->u));
  EXPECT_TRUE(
      std::holds_alternative<Fortran::parser::ErrorRecovery>(unit++->u));
  EXPECT_TRUE(std::holds_alternative<
      Fortran::common::Indirection<Fortran::parser::CompilerDirective>>(
      unit++->u));
  EXPECT_TRUE(std::holds_alternative<
      Fortran::common::Indirection<Fortran::parser::SubroutineSubprogram>>(
      unit->u));
}

TEST_F(FrontendActionTest, RecoveryNodesSupportTreeConsumers) {
  ASSERT_NO_FATAL_FAILURE(parseRecoveryNodes());
  auto &program = *compInst.getParsing().parseTree();

  RecoveryNodeVisitor visitor, mutator;
  const auto &constProgram = program;
  Fortran::parser::Walk(constProgram, visitor);
  Fortran::parser::Walk(program, mutator);
  for (const auto *counts : {&visitor, &mutator}) {
    EXPECT_EQ(counts->program, 1u);
    EXPECT_EQ(counts->internal, 1u);
    EXPECT_EQ(counts->module, 1u);
    EXPECT_EQ(counts->interface, 1u);
  }

  std::string dump;
  llvm::raw_string_ostream dumpStream{dump};
  Fortran::parser::DumpTree(dumpStream, constProgram);
  EXPECT_EQ(llvm::StringRef(dump).count("ErrorRecovery"), 4u);

  std::string unparsed;
  llvm::raw_string_ostream unparseStream{unparsed};
  Fortran::parser::Unparse(
      unparseStream, constProgram, compInst.getInvocation().getLangOpts());
  EXPECT_FALSE(llvm::StringRef(unparsed).contains("DECLARE"));
  for (const char *name : {"outer", "iface", "inner", "child"}) {
    EXPECT_TRUE(llvm::StringRef(unparsed).contains(name));
  }
}

TEST_F(FrontendActionTest, RecoveryNodesStopBeforeSemanticConsumers) {
  ASSERT_NO_FATAL_FAILURE(parseRecoveryNodes());
  llvm::InitializeAllAsmPrinters();
  for (auto action : {ParseSyntaxOnly, DebugDumpParseTreeNoSema,
           DebugUnparseNoSema, DebugMeasureParseTree, DebugDumpSymbols,
           DebugDumpPFT, EmitFIR, EmitLLVM}) {
    SCOPED_TRACE(static_cast<int>(action));
    CompilerInstance instance;
    auto &invocation = instance.getInvocation();
    invocation.getTargetOpts().triple =
        llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
    invocation.getFrontendOpts().features.Enable(
        Fortran::common::LanguageFeature::OpenACC);
    invocation.getFrontendOpts().inputs.emplace_back(
        inputFilePath, Language::Fortran);
    invocation.getFrontendOpts().programAction = action;
    instance.createDiagnostics();

    llvm::SmallVector<char> diagnostics, output;
    instance.setSemaOutputStream(
        std::make_unique<llvm::raw_svector_ostream>(diagnostics));
    instance.setOutputStream(
        std::make_unique<llvm::raw_svector_ostream>(output));
    EXPECT_FALSE(executeCompilerInvocation(&instance));
    EXPECT_TRUE(output.empty());
    EXPECT_TRUE(instance.getParsing().messages().AnyFatalError());
    EXPECT_TRUE(instance.getParsing().parseTree().has_value());
  }
}

TEST_F(FrontendActionTest, ModuleFileRecoveryIsRejected) {
  ASSERT_NO_FATAL_FAILURE(parseRecoveryNodes());
  CompilerInstance moduleInstance;
  auto &options = moduleInstance.getInvocation().getFortranOpts();
  options.features.Enable(Fortran::common::LanguageFeature::OpenACC);
  options.isModuleFile = true;
  auto &parsing = moduleInstance.getParsing();
  ASSERT_NE(parsing.Prescan(inputFilePath, options), nullptr);
  ASSERT_FALSE(parsing.messages().AnyFatalError());
  parsing.Parse(llvm::nulls(), moduleInstance.getInvocation().getLangOpts());

  // Module files defer messages. A recovered tree must not be mistaken for
  // a successful parse merely because it has no diagnostic messages.
  EXPECT_FALSE(parsing.parseTree().has_value());
}

// Building several PFTs must not change semantic symbols or another PFT.
TEST_F(FrontendActionTest, MainProgramGlobalsBelongToPFT) {
  *(inputFileOs) << "integer :: x,image,co[*]\n!$omp parallel\n"
                 << "print *,x,co[image]\n!$omp end parallel\nend\n";
  inputFileOs.reset();
  compInst.getFrontendOpts().features.Enable(
      Fortran::common::LanguageFeature::OpenMP);
  compInst.getFrontendOpts().programAction = ParseSyntaxOnly;
  ASSERT_TRUE(executeCompilerInvocation(&compInst));
  auto &semantics = compInst.getSemanticsContext();
  const auto &scope = semantics.globalScope().children().back();
  const auto *x = scope.FindSymbol(Fortran::parser::CharBlock{"x"});
  ASSERT_NE(x, nullptr);
  const auto flags = x->flags();
  const auto attrs = x->attrs();
  auto build = [&](bool enabled) {
    Fortran::lower::LoweringOptions options;
    options.setOpenMPMainProgramGlobals(enabled);
    return Fortran::lower::createPFT(
        *compInst.getParsing().parseTree(), semantics, options);
  };
  auto on = build(true), off = build(false), again = build(true);
  EXPECT_TRUE(x->flags() == flags && x->attrs() == attrs);
  EXPECT_TRUE(on->isMainGlobal(*x));
  EXPECT_FALSE(off->isMainGlobal(*x));
  EXPECT_TRUE(again->isMainGlobal(*x));
  EXPECT_FALSE(
      on->isMainGlobal(*scope.FindSymbol(Fortran::parser::CharBlock{"image"})));
  using Fortran::lower::pft::getDependentVariableList;
  EXPECT_TRUE(getDependentVariableList(*x, *on).back().isGlobal());
  EXPECT_FALSE(getDependentVariableList(*x, *off).back().isGlobal());
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
