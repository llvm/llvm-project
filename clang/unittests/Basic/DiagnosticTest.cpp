//===- unittests/Basic/DiagnosticTest.cpp -- Diagnostic engine tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticError.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticLex.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

using namespace llvm;
using namespace clang;

// Declare DiagnosticsTestHelper to avoid GCC warning
namespace clang {
void DiagnosticsTestHelper(DiagnosticsEngine &diag);
}

void clang::DiagnosticsTestHelper(DiagnosticsEngine &diag) {
  EXPECT_FALSE(diag.DiagStates.empty());
  EXPECT_TRUE(diag.DiagStatesByLoc.empty());
  EXPECT_TRUE(diag.DiagStateOnPushStack.empty());
}

namespace {
using testing::AllOf;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::IsEmpty;

MATCHER_P(WithMessage, M,
          "has diagnostic message that " +
              ::testing::DescribeMatcher<std::string>(M)) {
  return testing::ExplainMatchResult(M, arg.getMessage().str(),
                                     result_listener);
}
MATCHER(IsError, "has error severity") {
  return arg.getLevel() == DiagnosticsEngine::Level::Error;
}

// Check that DiagnosticErrorTrap works with SuppressAllDiagnostics.
TEST(DiagnosticTest, suppressAndTrap) {
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts,
                          new IgnoringDiagConsumer());
  Diags.setSuppressAllDiagnostics(true);

  {
    DiagnosticErrorTrap trap(Diags);

    // Diag that would set UncompilableErrorOccurred and ErrorOccurred.
    Diags.Report(diag::err_target_unknown_triple) << "unknown";

    // Diag that would set UnrecoverableErrorOccurred and ErrorOccurred.
    Diags.Report(diag::err_cannot_open_file) << "file" << "error";

    // Diag that would set FatalErrorOccurred
    // (via non-note following a fatal error).
    Diags.Report(diag::warn_apinotes_message) << "warning";

    EXPECT_TRUE(trap.hasErrorOccurred());
    EXPECT_TRUE(trap.hasUnrecoverableErrorOccurred());
  }

  EXPECT_FALSE(Diags.hasErrorOccurred());
  EXPECT_FALSE(Diags.hasFatalErrorOccurred());
  EXPECT_FALSE(Diags.hasUncompilableErrorOccurred());
  EXPECT_FALSE(Diags.hasUnrecoverableErrorOccurred());
}

// Check that FatalsAsError works as intended
TEST(DiagnosticTest, fatalsAsError) {
  for (unsigned FatalsAsError = 0; FatalsAsError != 2; ++FatalsAsError) {
    DiagnosticOptions DiagOpts;
    DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts,
                            new IgnoringDiagConsumer());
    Diags.setFatalsAsError(FatalsAsError);

    // Diag that would set UnrecoverableErrorOccurred and ErrorOccurred.
    Diags.Report(diag::err_cannot_open_file) << "file" << "error";

    // Diag that would set FatalErrorOccurred
    // (via non-note following a fatal error).
    Diags.Report(diag::warn_apinotes_message) << "warning";

    EXPECT_TRUE(Diags.hasErrorOccurred());
    EXPECT_EQ(Diags.hasFatalErrorOccurred(), FatalsAsError ? 0u : 1u);
    EXPECT_TRUE(Diags.hasUncompilableErrorOccurred());
    EXPECT_TRUE(Diags.hasUnrecoverableErrorOccurred());

    // The warning should be emitted and counted only if we're not suppressing
    // after fatal errors.
    EXPECT_EQ(Diags.getNumWarnings(), FatalsAsError);
  }
}

TEST(DiagnosticTest, tooManyErrorsIsAlwaysFatal) {
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts,
                          new IgnoringDiagConsumer());
  Diags.setFatalsAsError(true);

  // Report a fatal_too_many_errors diagnostic to ensure that still
  // acts as a fatal error despite downgrading fatal errors to errors.
  Diags.Report(diag::fatal_too_many_errors);
  EXPECT_TRUE(Diags.hasFatalErrorOccurred());

  // Ensure that the severity of that diagnostic is really "fatal".
  EXPECT_EQ(Diags.getDiagnosticLevel(diag::fatal_too_many_errors, {}),
            DiagnosticsEngine::Level::Fatal);
}

// Check that soft RESET works as intended
TEST(DiagnosticTest, softReset) {
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts,
                          new IgnoringDiagConsumer());

  unsigned numWarnings = 0U, numErrors = 0U;

  Diags.Reset(true);
  // Check For ErrorOccurred and TrapNumErrorsOccurred
  EXPECT_FALSE(Diags.hasErrorOccurred());
  EXPECT_FALSE(Diags.hasFatalErrorOccurred());
  EXPECT_FALSE(Diags.hasUncompilableErrorOccurred());
  // Check for UnrecoverableErrorOccurred and TrapNumUnrecoverableErrorsOccurred
  EXPECT_FALSE(Diags.hasUnrecoverableErrorOccurred());

  EXPECT_EQ(Diags.getNumWarnings(), numWarnings);
  EXPECT_EQ(Diags.getNumErrors(), numErrors);

  // Check for private variables of DiagnosticsEngine differentiating soft reset
  DiagnosticsTestHelper(Diags);

  EXPECT_TRUE(Diags.isLastDiagnosticIgnored());
}

TEST(DiagnosticTest, diagnosticError) {
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts,
                          new IgnoringDiagConsumer());
  PartialDiagnostic::DiagStorageAllocator Alloc;
  llvm::Expected<std::pair<int, int>> Value = DiagnosticError::create(
      SourceLocation(), PartialDiagnostic(diag::err_cannot_open_file, Alloc)
                            << "file"
                            << "error");
  ASSERT_TRUE(!Value);
  llvm::Error Err = Value.takeError();
  std::optional<PartialDiagnosticAt> ErrDiag = DiagnosticError::take(Err);
  llvm::cantFail(std::move(Err));
  ASSERT_FALSE(!ErrDiag);
  EXPECT_EQ(ErrDiag->first, SourceLocation());
  EXPECT_EQ(ErrDiag->second.getDiagID(), diag::err_cannot_open_file);

  Value = std::make_pair(20, 1);
  ASSERT_FALSE(!Value);
  EXPECT_EQ(*Value, std::make_pair(20, 1));
  EXPECT_EQ(Value->first, 20);
}

class CaptureDiagnosticConsumer : public DiagnosticConsumer {
public:
  SmallVector<StoredDiagnostic> StoredDiags;

  void HandleDiagnostic(DiagnosticsEngine::Level level,
                        const Diagnostic &Info) override {
    StoredDiags.push_back(StoredDiagnostic(level, Info));
  }
};

TEST(DiagnosticTest, storedDiagEmptyWarning) {
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts);

  CaptureDiagnosticConsumer CaptureConsumer;
  Diags.setClient(&CaptureConsumer, /*ShouldOwnClient=*/false);
  Diags.Report(diag::pp_hash_warning) << "";
  ASSERT_TRUE(CaptureConsumer.StoredDiags.size() == 1);

  // Make sure an empty warning can round-trip with \c StoredDiagnostic.
  Diags.Report(CaptureConsumer.StoredDiags.front());
}

// std::string_view is used by downstream consumers.
TEST(DiagnosticTest, reportAcceptsStringViewMessage) {
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts);

  CaptureDiagnosticConsumer CaptureConsumer;
  Diags.setClient(&CaptureConsumer, /*ShouldOwnClient=*/false);

  std::string_view SV = "diagnostic";
  Diags.Report(diag::err_target_unknown_triple) << SV;

  EXPECT_THAT(CaptureConsumer.StoredDiags,
              ElementsAre(WithMessage(HasSubstr("diagnostic"))));
}

class SuppressionMappingTest : public testing::Test {
public:
  SuppressionMappingTest() {
    Diags.setClient(&CaptureConsumer, /*ShouldOwnClient=*/false);
  }

protected:
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags{DiagnosticIDs::create(), DiagOpts};

  llvm::ArrayRef<StoredDiagnostic> diags() {
    return CaptureConsumer.StoredDiags;
  }

  SourceLocation locForFile(llvm::StringRef FileName) {
    auto Buf = MemoryBuffer::getMemBuffer("", FileName);
    SourceManager &SM = Diags.getSourceManager();
    FileID FooID = SM.createFileID(std::move(Buf));
    return SM.getLocForStartOfFile(FooID);
  }

private:
  FileManager FM{{}, FS};
  SourceManager SM{Diags, FM};

  class CaptureDiagnosticConsumer : public DiagnosticConsumer {
  public:
    std::vector<StoredDiagnostic> StoredDiags;

    void HandleDiagnostic(DiagnosticsEngine::Level level,
                          const Diagnostic &Info) override {
      StoredDiags.push_back(StoredDiagnostic(level, Info));
    }
  };
  CaptureDiagnosticConsumer CaptureConsumer;
};

TEST_F(SuppressionMappingTest, MissingMappingFile) {
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), ElementsAre(AllOf(
                           WithMessage("no such file or directory: 'foo.txt'"),
                           IsError())));
}

TEST_F(SuppressionMappingTest, MalformedFile) {
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer("asdf", "foo.txt"));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(),
              ElementsAre(AllOf(
                  WithMessage("failed to process suppression mapping file "
                              "'foo.txt': malformed line 1: 'asdf'"),
                  IsError())));
}

TEST_F(SuppressionMappingTest, UnknownDiagName) {
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer("[non-existing-warning]"));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), ElementsAre(WithMessage(
                           "unknown warning option 'non-existing-warning'")));
}

TEST_F(SuppressionMappingTest, SuppressesGroup) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());

  SourceLocation FooLoc = locForFile("foo.cpp");
  EXPECT_TRUE(Diags.isSuppressedViaMapping(diag::warn_unused_function, FooLoc));
  EXPECT_FALSE(Diags.isSuppressedViaMapping(diag::warn_deprecated, FooLoc));
}

TEST_F(SuppressionMappingTest, EmitCategoryIsExcluded) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*
  src:*foo.cpp=emit)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());

  EXPECT_TRUE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                           locForFile("bar.cpp")));
  EXPECT_FALSE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                            locForFile("foo.cpp")));
}

TEST_F(SuppressionMappingTest, LastMatchWins) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*clang/*
  src:*clang/lib/Sema/*=emit
  src:*clang/lib/Sema/foo*)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());

  EXPECT_TRUE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile("clang/lib/Basic/foo.h")));
  EXPECT_FALSE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile("clang/lib/Sema/bar.h")));
  EXPECT_TRUE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                           locForFile("clang/lib/Sema/foo.h")));
}

TEST_F(SuppressionMappingTest, LongShortMatch) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*test/*
  src:*lld/*=emit)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());

  EXPECT_TRUE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                           locForFile("test/t1.cpp")));
  EXPECT_FALSE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                            locForFile("lld/test/t2.cpp")));
}

TEST_F(SuppressionMappingTest, ShortLongMatch) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*lld/*=emit
  src:*test/*)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());

  EXPECT_TRUE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                           locForFile("test/t1.cpp")));
  EXPECT_TRUE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                           locForFile("lld/test/t2.cpp")));
}

TEST_F(SuppressionMappingTest, IsIgnored) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*clang/*)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  Diags.getDiagnosticOptions().Warnings = {"unused"};
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  ASSERT_THAT(diags(), IsEmpty());

  SourceManager &SM = Diags.getSourceManager();
  auto ClangID =
      SM.createFileID(llvm::MemoryBuffer::getMemBuffer("", "clang/foo.h"));
  auto NonClangID =
      SM.createFileID(llvm::MemoryBuffer::getMemBuffer("", "llvm/foo.h"));
  auto PresumedClangID =
      SM.createFileID(llvm::MemoryBuffer::getMemBuffer("", "llvm/foo2.h"));
  // Add a line directive to point into clang/foo.h
  SM.AddLineNote(SM.getLocForStartOfFile(PresumedClangID), 42,
                 SM.getLineTableFilenameID("clang/foo.h"), false, false,
                 clang::SrcMgr::C_User);

  EXPECT_TRUE(Diags.isIgnored(diag::warn_unused_function,
                              SM.getLocForStartOfFile(ClangID)));
  EXPECT_FALSE(Diags.isIgnored(diag::warn_unused_function,
                               SM.getLocForStartOfFile(NonClangID)));
  EXPECT_TRUE(Diags.isIgnored(diag::warn_unused_function,
                              SM.getLocForStartOfFile(PresumedClangID)));

  // Pretend we have a clang-diagnostic pragma to enforce the warning. Make sure
  // suppressing mapping doesn't take over.
  Diags.setSeverity(diag::warn_unused_function, diag::Severity::Error,
                    SM.getLocForStartOfFile(ClangID));
  EXPECT_FALSE(Diags.isIgnored(diag::warn_unused_function,
                               SM.getLocForStartOfFile(ClangID)));
}

TEST_F(SuppressionMappingTest, ParsingRespectsOtherWarningOpts) {
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer("[non-existing-warning]"));
  Diags.getDiagnosticOptions().Warnings.push_back("no-unknown-warning-option");
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());
}

#ifdef _WIN32
TEST_F(SuppressionMappingTest, CanonicalizesSlashesOnWindows) {
  llvm::StringLiteral SuppressionMappingFile = R"(#!special-case-list-v4
  [unused]
  src:*clang/*
  src:*clang/lib/Sema/*=emit
  src:*clang/lib\\Sema/foo*
  fun:suppress/me)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());

  EXPECT_TRUE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile(R"(clang/lib/Basic/bar.h)")));
  EXPECT_TRUE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile(R"(clang/lib/Basic\bar.h)")));
  EXPECT_TRUE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile(R"(clang\lib/Basic/bar.h)")));
  EXPECT_FALSE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile(R"(clang/lib/Sema/baz.h)")));
  EXPECT_FALSE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile(R"(clang/lib/Sema\baz.h)")));

  // Under slash-agnostic matching, backslashes and forward slashes match each
  // other, so we match the third pattern.
  EXPECT_TRUE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile(R"(clang\lib\Sema/foo.h)")));
  EXPECT_TRUE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile(R"(clang/lib/Sema/foo.h)")));
}
#endif

TEST(EscapeSingleCodepointForDiagnosticTest, printableDisplaysQuoted) {
  EXPECT_EQ(EscapeSingleCodepointForDiagnostic(U'A'), "'A'");
  // This test fails when msvc is not using /utf-8.
  // EXPECT_EQ(EscapeSingleCodepointForDiagnostic(U'🤡'), "'🤡' U+1F921");
  EXPECT_EQ(EscapeSingleCodepointForDiagnostic(U' '), "' '");
}

TEST(EscapeSingleCodepointForDiagnosticTest, nonPrintableDisplaysNoQuoted) {
  EXPECT_EQ(EscapeSingleCodepointForDiagnostic(U'\n'), "U+000A");
  EXPECT_EQ(EscapeSingleCodepointForDiagnostic(U'\0'), "U+0000");
  EXPECT_EQ(EscapeSingleCodepointForDiagnostic(U'\x1B'), "U+001B");
}

TEST(EscapeSingleCodepointForDiagnosticTest, nonScalarValues) {
  // Low and high surrogates:
  EXPECT_EQ(EscapeSingleCodepointForDiagnostic(0xD800), "<0xD800>");
  EXPECT_EQ(EscapeSingleCodepointForDiagnostic(0xDFFF), "<0xDFFF>");
  // Overly large values:
  EXPECT_EQ(EscapeSingleCodepointForDiagnostic(0x110000), "<0x110000>");
}

// Plugins register their diagnostics under a warning group they name at runtime
// (by convention "<plugin>-plugin"), so users can
// silence them with -Wno-<group> exactly like a built-in warning -- even though
// the group name is unknown to the compiler until the plugin is loaded.
class PluginWarningGroupTest : public testing::Test {
public:
  PluginWarningGroupTest() {
    Diags.setClient(&CaptureConsumer, /*ShouldOwnClient=*/false);
  }

protected:
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags{DiagnosticIDs::create(), DiagOpts};

  llvm::ArrayRef<StoredDiagnostic> diags() {
    return CaptureConsumer.StoredDiags;
  }

private:
  class CaptureDiagnosticConsumer : public DiagnosticConsumer {
  public:
    std::vector<StoredDiagnostic> StoredDiags;
    void HandleDiagnostic(DiagnosticsEngine::Level Level,
                          const Diagnostic &Info) override {
      StoredDiags.push_back(StoredDiagnostic(Level, Info));
    }
  };
  CaptureDiagnosticConsumer CaptureConsumer;
};

// -Wno-<group> silences a plugin diagnostic tagged with that group.
TEST_F(PluginWarningGroupTest, WnoSuppressesPluginGroup) {
  unsigned ID =
      Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                            "plugin reused an AST node", "example-plugin");
  EXPECT_FALSE(Diags.isIgnored(ID, SourceLocation()));

  DiagOpts.Warnings = {"no-example-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_TRUE(Diags.isIgnored(ID, SourceLocation()));
}

// The command line is parsed before the plugin loads, so the group name is
// still unknown when -Wno-<group> is processed. The mapping must be remembered
// and applied once the plugin registers the diagnostic -- and the flag must not
// be reported as an unknown warning option in the meantime.
TEST_F(PluginWarningGroupTest, WnoBeforePluginRegistersGroup) {
  DiagOpts.Warnings = {"no-example-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_THAT(diags(), IsEmpty());

  unsigned ID =
      Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                            "plugin reused an AST node", "example-plugin");
  EXPECT_TRUE(Diags.isIgnored(ID, SourceLocation()));
}

// -Wplugin is an umbrella over every plugin group.
TEST_F(PluginWarningGroupTest, WnoPluginUmbrellaSuppressesEveryPluginGroup) {
  DiagOpts.Warnings = {"no-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);

  unsigned First = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                                         "first diag", "example-plugin");
  unsigned Other = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                                         "other diag", "othertool-plugin");
  EXPECT_TRUE(Diags.isIgnored(First, SourceLocation()));
  EXPECT_TRUE(Diags.isIgnored(Other, SourceLocation()));
}

// -Wno-<group> only touches diagnostics in that group.
TEST_F(PluginWarningGroupTest, LeavesOtherDiagnosticsAlone) {
  unsigned Grouped = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                                           "grouped", "example-plugin");
  unsigned Ungrouped =
      Diags.getCustomDiagID(DiagnosticsEngine::Warning, "ungrouped");

  DiagOpts.Warnings = {"no-example-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_TRUE(Diags.isIgnored(Grouped, SourceLocation()));
  EXPECT_FALSE(Diags.isIgnored(Ungrouped, SourceLocation()));
}

// -Werror=<group> promotes a plugin warning to an error.
TEST_F(PluginWarningGroupTest, WerrorPromotesPluginGroup) {
  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Warning, "plugin diag",
                                      "example-plugin");
  DiagOpts.Warnings = {"error=example-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_EQ(Diags.getDiagnosticLevel(ID, SourceLocation()),
            DiagnosticsEngine::Error);
}

// Global -Werror promotes a plugin warning like any other warning.
TEST_F(PluginWarningGroupTest, GlobalWerrorPromotesPluginWarning) {
  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Warning, "plugin diag",
                                      "example-plugin");
  DiagOpts.Warnings = {"error"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_EQ(Diags.getDiagnosticLevel(ID, SourceLocation()),
            DiagnosticsEngine::Error);
}

// A -Wno-<x>-plugin naming a group that no plugin ever claims is deferred (not
// reported immediately as unknown), then flagged once all plugins have loaded.
TEST_F(PluginWarningGroupTest, UnclaimedPluginGroupReportedAfterLoad) {
  DiagOpts.Warnings = {"no-bogus-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_THAT(diags(), IsEmpty());

  Diags.getDiagnosticIDs()->reportUnclaimedPluginGroups(Diags);
  EXPECT_THAT(diags(), ElementsAre(WithMessage(
                           "unknown warning option '-Wbogus-plugin'")));
}

// A plugin group claimed by a loaded plugin is not reported, even though the
// -W flag named it before the plugin registered its diagnostic.
TEST_F(PluginWarningGroupTest, ClaimedPluginGroupNotReported) {
  DiagOpts.Warnings = {"no-example-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  Diags.getCustomDiagID(DiagnosticsEngine::Warning, "diag", "example-plugin");

  Diags.getDiagnosticIDs()->reportUnclaimedPluginGroups(Diags);
  EXPECT_THAT(diags(), IsEmpty());
}

// A diagnostic in a "<plugin>-plugin-<sub>" subgroup is silenced by its own
// name.
TEST_F(PluginWarningGroupTest, SubgroupSilencedByOwnName) {
  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Warning, "reuse",
                                      "example-plugin-loop");
  EXPECT_FALSE(Diags.isIgnored(ID, SourceLocation()));

  DiagOpts.Warnings = {"no-example-plugin-loop"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_TRUE(Diags.isIgnored(ID, SourceLocation()));
}

// A subgroup is also silenced by its parent "<plugin>-plugin" group, even when
// the flag is seen before the plugin registers the subgroup.
TEST_F(PluginWarningGroupTest, SubgroupSilencedByParentGroup) {
  DiagOpts.Warnings = {"no-example-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);

  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Warning, "reuse",
                                      "example-plugin-loop");
  EXPECT_TRUE(Diags.isIgnored(ID, SourceLocation()));
}

// -Wno-<subgroup> does not affect a sibling subgroup of the same plugin.
TEST_F(PluginWarningGroupTest, SiblingSubgroupUnaffected) {
  unsigned Reuse = Diags.getCustomDiagID(DiagnosticsEngine::Warning, "reuse",
                                         "example-plugin-loop");
  unsigned Other = Diags.getCustomDiagID(DiagnosticsEngine::Warning, "other",
                                         "example-plugin-cast");

  DiagOpts.Warnings = {"no-example-plugin-loop"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_TRUE(Diags.isIgnored(Reuse, SourceLocation()));
  EXPECT_FALSE(Diags.isIgnored(Other, SourceLocation()));
}

// When several controlling groups are set, the most specific one wins:
// -Wno-<parent> does not silence a subgroup a more specific -W<subgroup> keeps
// on.
TEST_F(PluginWarningGroupTest, MostSpecificGroupWins) {
  DiagOpts.Warnings = {"no-example-plugin", "example-plugin-loop"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);

  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Warning, "reuse",
                                      "example-plugin-loop");
  EXPECT_FALSE(Diags.isIgnored(ID, SourceLocation()));
}

// -Werror=plugin promotes every plugin diagnostic to an error.
TEST_F(PluginWarningGroupTest, WerrorUmbrellaPromotesAllPlugins) {
  DiagOpts.Warnings = {"error=plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);

  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Warning, "diag",
                                      "example-plugin-loop");
  EXPECT_EQ(Diags.getDiagnosticLevel(ID, SourceLocation()),
            DiagnosticsEngine::Error);
}

// A plugin that (unconventionally) names a built-in group joins that group
// directly, so the built-in flag controls it.
TEST_F(PluginWarningGroupTest, BuiltinGroupNameJoinsBuiltinGroup) {
  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                                      "unused thing", "unused");
  DiagOpts.Warnings = {"no-unused"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_TRUE(Diags.isIgnored(ID, SourceLocation()));
}

// A plugin remark is off by default and enabled by -R<group>; a -W flag on the
// same group does not touch it, so warnings, errors and remarks can share a
// group namespace.
TEST_F(PluginWarningGroupTest, RemarkGroupControlledByROption) {
  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Remark, "spent %0 ms",
                                      "example-plugin-perf");
  EXPECT_TRUE(Diags.isIgnored(ID, SourceLocation()));

  // A -W flag on the group does not enable the remark.
  DiagOpts.Warnings = {"example-plugin-perf"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_TRUE(Diags.isIgnored(ID, SourceLocation()));

  // -R<group> enables it.
  DiagOpts.Remarks = {"example-plugin-perf"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_FALSE(Diags.isIgnored(ID, SourceLocation()));
}

// A plugin error can be placed in a group too (for organization and the printed
// "[-W<group>]"), but a group flag must not silence it: -W/-R control warnings
// and remarks, never errors.
TEST_F(PluginWarningGroupTest, ErrorDiagnosticJoinsGroupButIsNotSuppressible) {
  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                      "unsupported '%0'", "example-plugin");
  EXPECT_EQ(Diags.getDiagnosticIDs()->getWarningOptionForDiag(ID),
            "example-plugin");

  DiagOpts.Warnings = {"no-example-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_FALSE(Diags.isIgnored(ID, SourceLocation()));
  EXPECT_EQ(Diags.getDiagnosticLevel(ID, SourceLocation()),
            DiagnosticsEngine::Error);
}

// Error protection also holds when the flag is seen before the plugin registers
// the error (exercises the mapping path, not the member-exclusion path above).
TEST_F(PluginWarningGroupTest,
       ErrorNotSuppressibleWhenFlagPrecedesRegistration) {
  DiagOpts.Warnings = {"no-example-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);

  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                      "unsupported '%0'", "example-plugin");
  EXPECT_EQ(Diags.getDiagnosticLevel(ID, SourceLocation()),
            DiagnosticsEngine::Error);
}

// A -R flag does not touch a warning in the same group (flavor separation).
TEST_F(PluginWarningGroupTest, RemarkFlagDoesNotAffectWarning) {
  unsigned ID =
      Diags.getCustomDiagID(DiagnosticsEngine::Warning, "w", "example-plugin");
  DiagOpts.Remarks = {"no-example-plugin"};
  ProcessWarningOptions(Diags, DiagOpts, *FS);
  EXPECT_FALSE(Diags.isIgnored(ID, SourceLocation()));
}
} // namespace
