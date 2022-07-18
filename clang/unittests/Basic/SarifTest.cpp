//===- unittests/Basic/SarifTest.cpp - Test writing SARIF documents -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Sarif.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>

using namespace clang;

namespace {

using LineCol = std::pair<unsigned int, unsigned int>;

static std::string serializeSarifDocument(llvm::json::Object &&Doc) {
  std::string Output;
  llvm::json::Value value(std::move(Doc));
  llvm::raw_string_ostream OS{Output};
  OS << llvm::formatv("{0}", value);
  OS.flush();
  return Output;
}

class SarifDocumentWriterTest : public ::testing::Test {
protected:
  SarifDocumentWriterTest()
      : InMemoryFileSystem(new llvm::vfs::InMemoryFileSystem),
        FileMgr(FileSystemOptions(), InMemoryFileSystem),
        DiagID(new DiagnosticIDs()), DiagOpts(new DiagnosticOptions()),
        Diags(DiagID, DiagOpts.get(), new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr) {}

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;

  FileID registerSource(llvm::StringRef Name, const char *SourceText,
                        bool IsMainFile = false) {
    std::unique_ptr<llvm::MemoryBuffer> SourceBuf =
        llvm::MemoryBuffer::getMemBuffer(SourceText);
    const FileEntry *SourceFile =
        FileMgr.getVirtualFile(Name, SourceBuf->getBufferSize(), 0);
    SourceMgr.overrideFileContents(SourceFile, std::move(SourceBuf));
    FileID FID = SourceMgr.getOrCreateFileID(SourceFile, SrcMgr::C_User);
    if (IsMainFile)
      SourceMgr.setMainFileID(FID);
    return FID;
  }

  CharSourceRange getFakeCharSourceRange(FileID FID, LineCol Begin,
                                         LineCol End) {
    auto BeginLoc = SourceMgr.translateLineCol(FID, Begin.first, Begin.second);
    auto EndLoc = SourceMgr.translateLineCol(FID, End.first, End.second);
    return CharSourceRange{SourceRange{BeginLoc, EndLoc}, /* ITR = */ false};
  }
};

TEST_F(SarifDocumentWriterTest, canCreateEmptyDocument) {
  // GIVEN:
  SarifDocumentWriter Writer{SourceMgr};

  // WHEN:
  const llvm::json::Object &EmptyDoc = Writer.createDocument();
  std::vector<StringRef> Keys(EmptyDoc.size());
  std::transform(EmptyDoc.begin(), EmptyDoc.end(), Keys.begin(),
                 [](auto item) { return item.getFirst(); });

  // THEN:
  ASSERT_THAT(Keys, testing::UnorderedElementsAre("$schema", "version"));
}

// Test that a newly inserted run will associate correct tool names
TEST_F(SarifDocumentWriterTest, canCreateDocumentWithOneRun) {
  // GIVEN:
  SarifDocumentWriter Writer{SourceMgr};
  const char *ShortName = "sariftest";
  const char *LongName = "sarif writer test";

  // WHEN:
  Writer.createRun(ShortName, LongName);
  Writer.endRun();
  const llvm::json::Object &Doc = Writer.createDocument();
  const llvm::json::Array *Runs = Doc.getArray("runs");

  // THEN:
  // A run was created
  ASSERT_THAT(Runs, testing::NotNull());

  // It is the only run
  ASSERT_EQ(Runs->size(), 1UL);

  // The tool associated with the run was the tool
  const llvm::json::Object *driver =
      Runs->begin()->getAsObject()->getObject("tool")->getObject("driver");
  ASSERT_THAT(driver, testing::NotNull());

  ASSERT_TRUE(driver->getString("name").hasValue());
  ASSERT_TRUE(driver->getString("fullName").hasValue());
  ASSERT_TRUE(driver->getString("language").hasValue());

  EXPECT_EQ(driver->getString("name").getValue(), ShortName);
  EXPECT_EQ(driver->getString("fullName").getValue(), LongName);
  EXPECT_EQ(driver->getString("language").getValue(), "en-US");
}

TEST_F(SarifDocumentWriterTest, addingResultsWillCrashIfThereIsNoRun) {
#if defined(NDEBUG) || !GTEST_HAS_DEATH_TEST
  GTEST_SKIP() << "This death test is only available for debug builds.";
#endif
  // GIVEN:
  SarifDocumentWriter Writer{SourceMgr};

  // WHEN:
  //  A SarifDocumentWriter::createRun(...) was not called prior to
  //  SarifDocumentWriter::appendResult(...)
  // But a rule exists
  auto RuleIdx = Writer.createRule(SarifRule::create());
  const SarifResult &EmptyResult = SarifResult::create(RuleIdx);

  // THEN:
  auto Matcher = ::testing::AnyOf(
      ::testing::HasSubstr("create a run first"),
      ::testing::HasSubstr("no runs associated with the document"));
  ASSERT_DEATH(Writer.appendResult(EmptyResult), Matcher);
}

// Test adding rule and result shows up in the final document
TEST_F(SarifDocumentWriterTest, addingResultWithValidRuleAndRunIsOk) {
  // GIVEN:
  SarifDocumentWriter Writer{SourceMgr};
  const SarifRule &Rule =
      SarifRule::create()
          .setRuleId("clang.unittest")
          .setDescription("Example rule created during unit tests")
          .setName("clang unit test");

  // WHEN:
  Writer.createRun("sarif test", "sarif test runner");
  unsigned RuleIdx = Writer.createRule(Rule);
  const SarifResult &result = SarifResult::create(RuleIdx);

  Writer.appendResult(result);
  const llvm::json::Object &Doc = Writer.createDocument();

  // THEN:
  // A document with a valid schema and version exists
  ASSERT_THAT(Doc.get("$schema"), ::testing::NotNull());
  ASSERT_THAT(Doc.get("version"), ::testing::NotNull());
  const llvm::json::Array *Runs = Doc.getArray("runs");

  // A run exists on this document
  ASSERT_THAT(Runs, ::testing::NotNull());
  ASSERT_EQ(Runs->size(), 1UL);
  const llvm::json::Object *TheRun = Runs->back().getAsObject();

  // The run has slots for tools, results, rules and artifacts
  ASSERT_THAT(TheRun->get("tool"), ::testing::NotNull());
  ASSERT_THAT(TheRun->get("results"), ::testing::NotNull());
  ASSERT_THAT(TheRun->get("artifacts"), ::testing::NotNull());
  const llvm::json::Object *Driver =
      TheRun->getObject("tool")->getObject("driver");
  const llvm::json::Array *Results = TheRun->getArray("results");
  const llvm::json::Array *Artifacts = TheRun->getArray("artifacts");

  // The tool is as expected
  ASSERT_TRUE(Driver->getString("name").hasValue());
  ASSERT_TRUE(Driver->getString("fullName").hasValue());

  EXPECT_EQ(Driver->getString("name").getValue(), "sarif test");
  EXPECT_EQ(Driver->getString("fullName").getValue(), "sarif test runner");

  // The results are as expected
  EXPECT_EQ(Results->size(), 1UL);

  // The artifacts are as expected
  EXPECT_TRUE(Artifacts->empty());
}

TEST_F(SarifDocumentWriterTest, checkSerializingResults) {
  // GIVEN:
  const std::string ExpectedOutput =
      R"({"$schema":"https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json","runs":[{"artifacts":[],"columnKind":"unicodeCodePoints","results":[{"message":{"text":""},"ruleId":"clang.unittest","ruleIndex":0}],"tool":{"driver":{"fullName":"sarif test runner","informationUri":"https://clang.llvm.org/docs/UsersManual.html","language":"en-US","name":"sarif test","rules":[{"fullDescription":{"text":"Example rule created during unit tests"},"id":"clang.unittest","name":"clang unit test"}],"version":"1.0.0"}}}],"version":"2.1.0"})";

  SarifDocumentWriter Writer{SourceMgr};
  const SarifRule &Rule =
      SarifRule::create()
          .setRuleId("clang.unittest")
          .setDescription("Example rule created during unit tests")
          .setName("clang unit test");

  // WHEN: A run contains a result
  Writer.createRun("sarif test", "sarif test runner", "1.0.0");
  unsigned ruleIdx = Writer.createRule(Rule);
  const SarifResult &Result = SarifResult::create(ruleIdx);
  Writer.appendResult(Result);
  std::string Output = serializeSarifDocument(Writer.createDocument());

  // THEN:
  ASSERT_THAT(Output, ::testing::StrEq(ExpectedOutput));
}

// Check that serializing artifacts from results produces valid SARIF
TEST_F(SarifDocumentWriterTest, checkSerializingArtifacts) {
  // GIVEN:
  const std::string ExpectedOutput =
      R"({"$schema":"https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json","runs":[{"artifacts":[{"length":40,"location":{"index":0,"uri":"file:///main.cpp"},"mimeType":"text/plain","roles":["resultFile"]}],"columnKind":"unicodeCodePoints","results":[{"locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":14,"startColumn":14,"startLine":3}}}],"message":{"text":"expected ';' after top level declarator"},"ruleId":"clang.unittest","ruleIndex":0}],"tool":{"driver":{"fullName":"sarif test runner","informationUri":"https://clang.llvm.org/docs/UsersManual.html","language":"en-US","name":"sarif test","rules":[{"fullDescription":{"text":"Example rule created during unit tests"},"id":"clang.unittest","name":"clang unit test"}],"version":"1.0.0"}}}],"version":"2.1.0"})";

  SarifDocumentWriter Writer{SourceMgr};
  const SarifRule &Rule =
      SarifRule::create()
          .setRuleId("clang.unittest")
          .setDescription("Example rule created during unit tests")
          .setName("clang unit test");

  // WHEN: A result is added with valid source locations for its diagnostics
  Writer.createRun("sarif test", "sarif test runner", "1.0.0");
  unsigned RuleIdx = Writer.createRule(Rule);

  llvm::SmallVector<CharSourceRange, 1> DiagLocs;
  const char *SourceText = "int foo = 0;\n"
                           "int bar = 1;\n"
                           "float x = 0.0\n";

  FileID MainFileID =
      registerSource("/main.cpp", SourceText, /* IsMainFile = */ true);
  CharSourceRange SourceCSR =
      getFakeCharSourceRange(MainFileID, {3, 14}, {3, 14});

  DiagLocs.push_back(SourceCSR);

  const SarifResult &Result =
      SarifResult::create(RuleIdx).setLocations(DiagLocs).setDiagnosticMessage(
          "expected ';' after top level declarator");
  Writer.appendResult(Result);
  std::string Output = serializeSarifDocument(Writer.createDocument());

  // THEN: Assert that the serialized SARIF is as expected
  ASSERT_THAT(Output, ::testing::StrEq(ExpectedOutput));
}

TEST_F(SarifDocumentWriterTest, checkSerializingCodeflows) {
  // GIVEN:
  const std::string ExpectedOutput =
      R"({"$schema":"https://docs.oasis-open.org/sarif/sarif/v2.1.0/cos02/schemas/sarif-schema-2.1.0.json","runs":[{"artifacts":[{"length":27,"location":{"index":1,"uri":"file:///test-header-1.h"},"mimeType":"text/plain","roles":["resultFile"]},{"length":30,"location":{"index":2,"uri":"file:///test-header-2.h"},"mimeType":"text/plain","roles":["resultFile"]},{"length":28,"location":{"index":3,"uri":"file:///test-header-3.h"},"mimeType":"text/plain","roles":["resultFile"]},{"length":41,"location":{"index":0,"uri":"file:///main.cpp"},"mimeType":"text/plain","roles":["resultFile"]}],"columnKind":"unicodeCodePoints","results":[{"codeFlows":[{"threadFlows":[{"locations":[{"importance":"essential","location":{"message":{"text":"Message #1"},"physicalLocation":{"artifactLocation":{"index":1},"region":{"endColumn":8,"endLine":2,"startColumn":1,"startLine":1}}}},{"importance":"important","location":{"message":{"text":"Message #2"},"physicalLocation":{"artifactLocation":{"index":2},"region":{"endColumn":8,"endLine":2,"startColumn":1,"startLine":1}}}},{"importance":"unimportant","location":{"message":{"text":"Message #3"},"physicalLocation":{"artifactLocation":{"index":3},"region":{"endColumn":8,"endLine":2,"startColumn":1,"startLine":1}}}}]}]}],"locations":[{"physicalLocation":{"artifactLocation":{"index":0},"region":{"endColumn":8,"endLine":2,"startColumn":5,"startLine":2}}}],"message":{"text":"Redefinition of 'foo'"},"ruleId":"clang.unittest","ruleIndex":0}],"tool":{"driver":{"fullName":"sarif test runner","informationUri":"https://clang.llvm.org/docs/UsersManual.html","language":"en-US","name":"sarif test","rules":[{"fullDescription":{"text":"Example rule created during unit tests"},"id":"clang.unittest","name":"clang unit test"}],"version":"1.0.0"}}}],"version":"2.1.0"})";

  const char *SourceText = "int foo = 0;\n"
                           "int foo = 1;\n"
                           "float x = 0.0;\n";
  FileID MainFileID =
      registerSource("/main.cpp", SourceText, /* IsMainFile = */ true);
  CharSourceRange DiagLoc{getFakeCharSourceRange(MainFileID, {2, 5}, {2, 8})};

  SarifDocumentWriter Writer{SourceMgr};
  const SarifRule &Rule =
      SarifRule::create()
          .setRuleId("clang.unittest")
          .setDescription("Example rule created during unit tests")
          .setName("clang unit test");

  constexpr unsigned int NUM_CASES = 3;
  llvm::SmallVector<ThreadFlow, NUM_CASES> Threadflows;
  const char *HeaderTexts[NUM_CASES]{("#pragma once\n"
                                      "#include <foo>"),
                                     ("#ifndef FOO\n"
                                      "#define FOO\n"
                                      "#endif"),
                                     ("#ifdef FOO\n"
                                      "#undef FOO\n"
                                      "#endif")};
  const char *HeaderNames[NUM_CASES]{"/test-header-1.h", "/test-header-2.h",
                                     "/test-header-3.h"};
  ThreadFlowImportance Importances[NUM_CASES]{
      ThreadFlowImportance::Essential, ThreadFlowImportance::Important,
      ThreadFlowImportance::Unimportant};
  for (size_t Idx = 0; Idx != NUM_CASES; ++Idx) {
    FileID FID = registerSource(HeaderNames[Idx], HeaderTexts[Idx]);
    CharSourceRange &&CSR = getFakeCharSourceRange(FID, {1, 1}, {2, 8});
    std::string Message = llvm::formatv("Message #{0}", Idx + 1);
    ThreadFlow Item = ThreadFlow::create()
                          .setRange(CSR)
                          .setImportance(Importances[Idx])
                          .setMessage(Message);
    Threadflows.push_back(Item);
  }

  // WHEN: A result containing code flows and diagnostic locations is added
  Writer.createRun("sarif test", "sarif test runner", "1.0.0");
  unsigned RuleIdx = Writer.createRule(Rule);
  const SarifResult &Result = SarifResult::create(RuleIdx)
                                  .setLocations({DiagLoc})
                                  .setDiagnosticMessage("Redefinition of 'foo'")
                                  .setThreadFlows(Threadflows);
  Writer.appendResult(Result);
  std::string Output = serializeSarifDocument(Writer.createDocument());

  // THEN: Assert that the serialized SARIF is as expected
  ASSERT_THAT(Output, ::testing::StrEq(ExpectedOutput));
}

} // namespace
