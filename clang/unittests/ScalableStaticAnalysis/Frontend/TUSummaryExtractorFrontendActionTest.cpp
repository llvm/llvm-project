//===- TUSummaryExtractorFrontendActionTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Frontend/TUSummaryExtractorFrontendAction.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormat.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/SerializationFormatRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>

using namespace clang;
using namespace ssaf;
using ::testing::Contains;
using ::testing::UnorderedElementsAre;

static auto errorsMsgsOf(const TextDiagnosticBuffer &Diags) {
  auto Errors = llvm::make_range(Diags.err_begin(), Diags.err_end());
  return llvm::make_second_range(Errors);
}
namespace {

/// A no-op TUSummaryExtractor suitable for use with a real TUSummaryBuilder.
class NoOpExtractor : public TUSummaryExtractor {
public:
  using TUSummaryExtractor::TUSummaryExtractor;
  void HandleTranslationUnit(ASTContext &Ctx) override {}
};
} // namespace

static TUSummaryExtractorRegistry::Add<NoOpExtractor>
    RegisterNoOp("NoOpExtractor", "No-op extractor for frontend action tests");

namespace {
class FailingSerializationFormat final : public SerializationFormat {
public:
  static llvm::Error failing(llvm::StringRef Component) {
    return llvm::createStringError(
        "error from always failing serialization format: " + Component);
  }

  llvm::Expected<TUSummary> readTUSummary(llvm::StringRef Path) override {
    return failing("readTUSummary");
  }

  llvm::Error writeTUSummary(const TUSummary &Summary,
                             llvm::StringRef Path) override {
    return failing("writeTUSummary");
  }

  llvm::Expected<TUSummaryEncoding>
  readTUSummaryEncoding(llvm::StringRef Path) override {
    return failing("readTUSummaryEncoding");
  }

  llvm::Error writeTUSummaryEncoding(const TUSummaryEncoding &SummaryEncoding,
                                     llvm::StringRef Path) override {
    return failing("writeTUSummaryEncoding");
  }

  llvm::Expected<LUSummary> readLUSummary(llvm::StringRef Path) override {
    return failing("readLUSummary");
  }

  llvm::Error writeLUSummary(const LUSummary &Summary,
                             llvm::StringRef Path) override {
    return failing("writeLUSummary");
  }

  llvm::Expected<LUSummaryEncoding>
  readLUSummaryEncoding(llvm::StringRef Path) override {
    return failing("readLUSummaryEncoding");
  }

  llvm::Error writeLUSummaryEncoding(const LUSummaryEncoding &SummaryEncoding,
                                     llvm::StringRef Path) override {
    return failing("writeLUSummaryEncoding");
  }

  llvm::Expected<StaticLibrary>
  readStaticLibrary(llvm::StringRef Path) override {
    return failing("readStaticLibrary");
  }

  llvm::Error writeStaticLibrary(const StaticLibrary &S,
                                 llvm::StringRef Path) override {
    return failing("writeStaticLibrary");
  }

  llvm::Expected<MultiArchStaticLibrary>
  readMultiArchStaticLibrary(llvm::StringRef Path) override {
    return failing("readMultiArchStaticLibrary");
  }

  llvm::Error writeMultiArchStaticLibrary(const MultiArchStaticLibrary &M,
                                          llvm::StringRef Path) override {
    return failing("writeMultiArchStaticLibrary");
  }

  llvm::Expected<MultiArchSharedLibrary>
  readMultiArchSharedLibrary(llvm::StringRef Path) override {
    return failing("readMultiArchSharedLibrary");
  }

  llvm::Error writeMultiArchSharedLibrary(const MultiArchSharedLibrary &M,
                                          llvm::StringRef Path) override {
    return failing("writeMultiArchSharedLibrary");
  }

  llvm::Expected<WPASuite> readWPASuite(llvm::StringRef Path) override {
    return failing("readWPASuite");
  }

  llvm::Error writeWPASuite(const WPASuite &Suite,
                            llvm::StringRef Path) override {
    return failing("writeWPASuite");
  }

  llvm::Expected<Artifact> readArtifact(llvm::StringRef Path) override {
    return failing("readArtifact");
  }

  llvm::Error writeArtifact(const Artifact &A, llvm::StringRef Path) override {
    return failing("writeArtifact");
  }

  llvm::Expected<ArtifactEncoding>
  readArtifactEncoding(llvm::StringRef Path) override {
    return failing("readArtifactEncoding");
  }

  llvm::Error writeArtifactEncoding(const ArtifactEncoding &E,
                                    llvm::StringRef Path) override {
    return failing("writeArtifactEncoding");
  }

  void forEachRegisteredAnalysis(
      llvm::function_ref<void(llvm::StringRef Name, llvm::StringRef Desc)>
          Callback) const override {}
};
} // namespace

static SerializationFormatRegistry::Add<FailingSerializationFormat>
    RegisterFormat(
        "FailingSerializationFormat",
        "A serialization format that fails on every possible operation.");

namespace {
/// Records the \c CompilationUnit namespace name of the most recently written
/// \c TUSummary into a static, so tests can assert that the runner threads
/// the configured CU name verbatim into the produced summary.
class CapturingSerializationFormat final : public SerializationFormat {
public:
  static std::string &lastCapturedName() {
    static std::string Name;
    return Name;
  }

  static void resetCapturedName() { lastCapturedName().clear(); }

  llvm::Expected<TUSummary> readTUSummary(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeTUSummary(const TUSummary &Summary,
                             llvm::StringRef) override {
    lastCapturedName() = getName(getTUNamespace(Summary));
    return llvm::Error::success();
  }
  llvm::Expected<TUSummaryEncoding>
  readTUSummaryEncoding(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeTUSummaryEncoding(const TUSummaryEncoding &,
                                     llvm::StringRef) override {
    return llvm::Error::success();
  }
  llvm::Expected<LUSummary> readLUSummary(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeLUSummary(const LUSummary &, llvm::StringRef) override {
    return llvm::Error::success();
  }
  llvm::Expected<LUSummaryEncoding>
  readLUSummaryEncoding(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeLUSummaryEncoding(const LUSummaryEncoding &,
                                     llvm::StringRef) override {
    return llvm::Error::success();
  }
  llvm::Expected<StaticLibrary> readStaticLibrary(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeStaticLibrary(const StaticLibrary &,
                                 llvm::StringRef) override {
    return llvm::Error::success();
  }
  llvm::Expected<MultiArchStaticLibrary>
  readMultiArchStaticLibrary(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeMultiArchStaticLibrary(const MultiArchStaticLibrary &,
                                          llvm::StringRef) override {
    return llvm::Error::success();
  }
  llvm::Expected<MultiArchSharedLibrary>
  readMultiArchSharedLibrary(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeMultiArchSharedLibrary(const MultiArchSharedLibrary &,
                                          llvm::StringRef) override {
    return llvm::Error::success();
  }
  llvm::Expected<WPASuite> readWPASuite(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeWPASuite(const WPASuite &, llvm::StringRef) override {
    return llvm::Error::success();
  }
  llvm::Expected<Artifact> readArtifact(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeArtifact(const Artifact &, llvm::StringRef) override {
    return llvm::Error::success();
  }
  llvm::Expected<ArtifactEncoding>
  readArtifactEncoding(llvm::StringRef) override {
    return llvm::createStringError("not implemented");
  }
  llvm::Error writeArtifactEncoding(const ArtifactEncoding &,
                                    llvm::StringRef) override {
    return llvm::Error::success();
  }
  void forEachRegisteredAnalysis(
      llvm::function_ref<void(llvm::StringRef, llvm::StringRef)>)
      const override {}
};
} // namespace

static SerializationFormatRegistry::Add<CapturingSerializationFormat>
    RegisterCapturingFormat(
        "CapturingSerializationFormat",
        "A serialization format that captures the CU namespace name.");

using EventLog = std::vector<std::string>;

namespace {

/// An ASTConsumer that logs callback invocations into a shared log.
class RecordingASTConsumer : public ASTConsumer {
public:
  RecordingASTConsumer(EventLog &Log, std::string Tag)
      : Log(Log), Tag(std::move(Tag)) {}

  void Initialize(ASTContext &Ctx) override {
    Log.push_back(Tag + "::Initialize");
  }
  bool HandleTopLevelDecl(DeclGroupRef D) override {
    Log.push_back(Tag + "::HandleTopLevelDecl");
    return true;
  }
  void HandleTranslationUnit(ASTContext &Ctx) override {
    Log.push_back(Tag + "::HandleTranslationUnit");
  }

private:
  EventLog &Log;
  std::string Tag;
};

/// A FrontendAction that returns a RecordingASTConsumer with the tag "Wrapped".
class RecordingAction : public ASTFrontendAction {
public:
  EventLog &getLog() { return Log; }
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &,
                                                 StringRef) override {
    return std::make_unique<RecordingASTConsumer>(Log, /*Tag=*/"Wrapped");
  }

private:
  EventLog Log;
};

class FailingAction : public ASTFrontendAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &,
                                                 StringRef) override {
    return nullptr;
  }
};

/// Creates a CompilerInstance configured with an in-memory "test.cc" file
/// containing "int x = 42;".
static std::unique_ptr<CompilerInstance>
makeCompiler(TextDiagnosticBuffer &DiagBuf) {
  auto Invocation = std::make_shared<CompilerInvocation>();
  Invocation->getPreprocessorOpts().addRemappedFile(
      "test.cc", llvm::MemoryBuffer::getMemBuffer("int x = 42;").release());
  Invocation->getFrontendOpts().Inputs.push_back(
      FrontendInputFile("test.cc", Language::CXX));
  Invocation->getFrontendOpts().ProgramAction = frontend::ParseSyntaxOnly;
  Invocation->getTargetOpts().Triple = "i386-unknown-linux-gnu";
  auto Compiler = std::make_unique<CompilerInstance>(std::move(Invocation));
  Compiler->setVirtualFileSystem(llvm::vfs::getRealFileSystem());
  Compiler->createDiagnostics(&DiagBuf, /*ShouldOwnClient=*/false);
  return Compiler;
}

struct TUSummaryExtractorFrontendActionTest : testing::Test {
  using PathString = llvm::SmallString<128>;
  PathString TestDir;
  TextDiagnosticBuffer DiagBuf;
  std::unique_ptr<CompilerInstance> Compiler = makeCompiler(DiagBuf);

  void SetUp() override {
    std::error_code EC = llvm::sys::fs::createUniqueDirectory(
        "ssaf-frontend-action-test", TestDir);
    ASSERT_FALSE(EC) << "Failed to create temp directory: " << EC.message();
  }

  void TearDown() override { llvm::sys::fs::remove_directories(TestDir); }

  std::string makePath(llvm::StringRef FileOrDirectoryName) const {
    PathString FullPath = TestDir;
    llvm::sys::path::append(FullPath, FileOrDirectoryName);
    return FullPath.str().str();
  }
};

TEST_F(TUSummaryExtractorFrontendActionTest,
       WrappedActionFailsToCreateConsumer) {
  // Configure valid SSAF options so the failure is purely from the wrapped
  // action, not from runner creation.
  std::string Output = makePath("output.MockSerializationFormat");
  Compiler->getSSAFOpts().TUSummaryFile = Output;
  Compiler->getSSAFOpts().ExtractSummaries = {"NoOpExtractor"};
  Compiler->getSSAFOpts().CompilationUnitId = "test-cu";

  TUSummaryExtractorFrontendAction ExtractorAction(
      std::make_unique<FailingAction>());
  Compiler->ExecuteAction(ExtractorAction);

  // If the wrapped action fails, the ExtractorAction should not output.
  EXPECT_FALSE(llvm::sys::fs::exists(Output));
}

TEST_F(TUSummaryExtractorFrontendActionTest,
       RunnerFailsWithInvalidFormat_WrappedConsumerStillRuns) {
  // Use an unregistered format extension so TUSummaryRunner::create fails.
  std::string Output = makePath("output.xyz");
  Compiler->getSSAFOpts().TUSummaryFile = Output;
  Compiler->getSSAFOpts().ExtractSummaries = {"NoOpExtractor"};
  Compiler->getSSAFOpts().CompilationUnitId = "test-cu";

  auto Wrapped = std::make_unique<RecordingAction>();
  const EventLog &Log = Wrapped->getLog();
  TUSummaryExtractorFrontendAction ExtractorAction(std::move(Wrapped));

  // The runner fails, so ExecuteAction should return false due to the fatal
  // diagnostic.
  EXPECT_FALSE(Compiler->ExecuteAction(ExtractorAction));

  // The wrapped consumer should still have run.
  EXPECT_THAT(Log, Contains("Wrapped::Initialize"));
  EXPECT_THAT(Log, Contains("Wrapped::HandleTranslationUnit"));

  // Exactly one error about the unknown format.
  EXPECT_THAT(errorsMsgsOf(DiagBuf),
              UnorderedElementsAre(
                  "unknown output summary file format 'xyz' specified by "
                  "'--ssaf-tu-summary-file=" +
                  Output + "'"));

  // No output should have been created due to the failure.
  EXPECT_FALSE(llvm::sys::fs::exists(Output));
}

TEST_F(TUSummaryExtractorFrontendActionTest,
       RunnerFailsWithUnknownExtractor_WrappedConsumerStillRuns) {
  std::string Output = makePath("output.MockSerializationFormat");
  Compiler->getSSAFOpts().TUSummaryFile = Output;
  Compiler->getSSAFOpts().ExtractSummaries = {"NonExistentExtractor"};
  Compiler->getSSAFOpts().CompilationUnitId = "test-cu";

  auto Wrapped = std::make_unique<RecordingAction>();
  const EventLog &Log = Wrapped->getLog();
  TUSummaryExtractorFrontendAction ExtractorAction(std::move(Wrapped));
  EXPECT_FALSE(Compiler->ExecuteAction(ExtractorAction));

  // The wrapped consumer should still have run.
  EXPECT_THAT(Log, Contains("Wrapped::Initialize"));
  EXPECT_THAT(Log, Contains("Wrapped::HandleTranslationUnit"));

  // Exactly one error about the unknown extractor.
  EXPECT_THAT(errorsMsgsOf(DiagBuf),
              UnorderedElementsAre("no summary extractor was registered with "
                                   "name: NonExistentExtractor"));

  // No output should have been created due to the failure.
  EXPECT_FALSE(llvm::sys::fs::exists(Output));
}

TEST_F(TUSummaryExtractorFrontendActionTest,
       RunnerSucceeds_ASTConsumerCallbacksPropagate) {
  std::string Output = makePath("output.MockSerializationFormat");
  Compiler->getSSAFOpts().TUSummaryFile = Output;
  Compiler->getSSAFOpts().ExtractSummaries = {"NoOpExtractor"};
  Compiler->getSSAFOpts().CompilationUnitId = "test-cu";

  auto Wrapped = std::make_unique<RecordingAction>();
  const EventLog &Log = Wrapped->getLog();
  TUSummaryExtractorFrontendAction ExtractorAction(std::move(Wrapped));
  EXPECT_TRUE(Compiler->ExecuteAction(ExtractorAction));

  // All wrapped ASTConsumer callbacks should have fired, not just
  // HandleTranslationUnit.
  EXPECT_THAT(Log, Contains("Wrapped::Initialize"));
  EXPECT_THAT(Log, Contains("Wrapped::HandleTopLevelDecl"));
  EXPECT_THAT(Log, Contains("Wrapped::HandleTranslationUnit"));
  EXPECT_EQ(DiagBuf.getNumErrors(), 0U);

  // The runner should have written output.
  EXPECT_TRUE(llvm::sys::fs::exists(Output));
}

// Use a custom action that checks whether the output path exists during
// HandleTranslationUnit — it should not, because the wrapped consumer runs
// before the runner.
struct OrderCheckingAction : public ASTFrontendAction {
  EventLog Log;
  std::string OutputPath;

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    struct Consumer : public ASTConsumer {
      Consumer(EventLog &Log, std::string OutputPath)
          : Log(Log), OutputPath(std::move(OutputPath)) {}
      void Initialize(ASTContext &) override {
        Log.push_back("Wrapped::Initialize");
      }
      bool HandleTopLevelDecl(DeclGroupRef) override {
        Log.push_back("Wrapped::HandleTopLevelDecl");
        return true;
      }
      void HandleTranslationUnit(ASTContext &) override {
        bool Exists = llvm::sys::fs::exists(OutputPath);
        Log.push_back(std::string("OutputExistsDuringWrappedHTU=") +
                      (Exists ? "true" : "false"));
        Log.push_back("Wrapped::HandleTranslationUnit");
      }

      EventLog &Log;
      std::string OutputPath;
    };
    return std::make_unique<Consumer>(Log, OutputPath);
  }
};
TEST_F(TUSummaryExtractorFrontendActionTest,
       RunnerSucceeds_WrappedRunsBeforeRunner) {
  std::string Output = makePath("output.MockSerializationFormat");
  Compiler->getSSAFOpts().TUSummaryFile = Output;
  Compiler->getSSAFOpts().ExtractSummaries = {"NoOpExtractor"};
  Compiler->getSSAFOpts().CompilationUnitId = "test-cu";

  auto Wrapped = std::make_unique<OrderCheckingAction>();
  Wrapped->OutputPath = Output;
  const EventLog &Log = Wrapped->Log;
  TUSummaryExtractorFrontendAction Action(std::move(Wrapped));

  EXPECT_TRUE(Compiler->ExecuteAction(Action));
  EXPECT_EQ(DiagBuf.getNumErrors(), 0U);

  // The output should NOT have existed when the wrapped consumer's
  // HandleTranslationUnit ran (wrapped is at index 0, runner at index 1).
  EXPECT_THAT(Log, Contains("OutputExistsDuringWrappedHTU=false"));

  // After ExecuteAction, the output should exist.
  EXPECT_TRUE(llvm::sys::fs::exists(Output));
}

TEST_F(TUSummaryExtractorFrontendActionTest, RunnerFailsToWrite) {
  std::string Output = makePath("output.FailingSerializationFormat");
  Compiler->getSSAFOpts().TUSummaryFile = Output;
  Compiler->getSSAFOpts().ExtractSummaries = {"NoOpExtractor"};
  Compiler->getSSAFOpts().CompilationUnitId = "test-cu";

  TUSummaryExtractorFrontendAction Action(std::make_unique<RecordingAction>());

  // This should fail because the summary writing fails and emits an error
  // diagnostic.
  EXPECT_FALSE(Compiler->ExecuteAction(Action));
  EXPECT_THAT(
      errorsMsgsOf(DiagBuf),
      UnorderedElementsAre(
          "failed to write TU summary to '" + Output +
          "': error from always failing serialization format: writeTUSummary"));

  // No output should have been created due to the failure.
  EXPECT_FALSE(llvm::sys::fs::exists(Output));
}

TEST_F(TUSummaryExtractorFrontendActionTest,
       MissingCompilationUnitIdDiagnoses) {
  std::string Output = makePath("output.MockSerializationFormat");
  Compiler->getSSAFOpts().TUSummaryFile = Output;
  Compiler->getSSAFOpts().ExtractSummaries = {"NoOpExtractor"};
  // SSAFCompilationUnitId left empty.

  auto Wrapped = std::make_unique<RecordingAction>();
  const EventLog &Log = Wrapped->getLog();
  TUSummaryExtractorFrontendAction ExtractorAction(std::move(Wrapped));

  EXPECT_FALSE(Compiler->ExecuteAction(ExtractorAction));

  EXPECT_THAT(Log, Contains("Wrapped::Initialize"));
  EXPECT_THAT(Log, Contains("Wrapped::HandleTranslationUnit"));

  EXPECT_THAT(errorsMsgsOf(DiagBuf),
              UnorderedElementsAre("option '--ssaf-tu-summary-file=' requires "
                                   "'--ssaf-compilation-unit-id=' to be set"));

  EXPECT_FALSE(llvm::sys::fs::exists(Output));
}

TEST_F(TUSummaryExtractorFrontendActionTest, EmptyCompilationUnitIdDiagnoses) {
  std::string Output = makePath("output.MockSerializationFormat");
  Compiler->getSSAFOpts().TUSummaryFile = Output;
  Compiler->getSSAFOpts().ExtractSummaries = {"NoOpExtractor"};
  Compiler->getSSAFOpts().CompilationUnitId = "";

  auto Wrapped = std::make_unique<RecordingAction>();
  const EventLog &Log = Wrapped->getLog();
  TUSummaryExtractorFrontendAction ExtractorAction(std::move(Wrapped));

  EXPECT_FALSE(Compiler->ExecuteAction(ExtractorAction));

  EXPECT_THAT(Log, Contains("Wrapped::Initialize"));
  EXPECT_THAT(Log, Contains("Wrapped::HandleTranslationUnit"));

  EXPECT_THAT(errorsMsgsOf(DiagBuf),
              UnorderedElementsAre("option '--ssaf-tu-summary-file=' requires "
                                   "'--ssaf-compilation-unit-id=' to be set"));

  EXPECT_FALSE(llvm::sys::fs::exists(Output));
}

TEST_F(TUSummaryExtractorFrontendActionTest,
       CompilationUnitIdThreadsToSummary) {
  CapturingSerializationFormat::resetCapturedName();

  const std::string CUId = "cu-X-test";
  std::string Output = makePath("output.CapturingSerializationFormat");
  Compiler->getSSAFOpts().TUSummaryFile = Output;
  Compiler->getSSAFOpts().ExtractSummaries = {"NoOpExtractor"};
  Compiler->getSSAFOpts().CompilationUnitId = CUId;

  TUSummaryExtractorFrontendAction Action(std::make_unique<RecordingAction>());
  EXPECT_TRUE(Compiler->ExecuteAction(Action));
  EXPECT_EQ(DiagBuf.getNumErrors(), 0U);

  EXPECT_EQ(CapturingSerializationFormat::lastCapturedName(), CUId);
}

} // namespace
