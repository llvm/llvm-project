//===- SARIFFormatTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/Sarif.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Version.h"
#include "clang/ScalableStaticAnalysis/SourceTransformation/SARIFTransformationReportFormat.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;
using namespace ssaf;

namespace {

struct TempPath {
  SmallString<128> Path;

  TempPath(StringRef Suffix) {
    sys::fs::createUniquePath("ssaf-sarif-%%%%%%." + Suffix, Path,
                              /*MakeAbsolute=*/true);
  }
  ~TempPath() { sys::fs::remove(Path); }
};

class SARIFFormatTest : public ::testing::Test {
protected:
  SARIFFormatTest()
      : Diags(DiagnosticIDs::create(), DiagOpts, new IgnoringDiagConsumer()),
        VFS(makeIntrusiveRefCnt<vfs::InMemoryFileSystem>()),
        FileMgr(FileSystemOptions(), VFS), SourceMgr(Diags, FileMgr) {}

  json::Value writeAndParse(const ReportDocument &Doc) {
    TempPath TP("sarif");
    EXPECT_THAT_ERROR(writeSARIFTransformationReport(Doc, TP.Path),
                      Succeeded());
    auto BufferOrErr = MemoryBuffer::getFile(TP.Path);
    EXPECT_TRUE(static_cast<bool>(BufferOrErr));
    auto ParsedOrErr = json::parse((*BufferOrErr)->getBuffer());
    EXPECT_THAT_EXPECTED(ParsedOrErr, Succeeded());
    return std::move(*ParsedOrErr);
  }

  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags;
  IntrusiveRefCntPtr<vfs::InMemoryFileSystem> VFS;
  FileManager FileMgr;
  SourceManager SourceMgr;
};

TEST_F(SARIFFormatTest, ToolDriverNameAndVersion) {
  ReportDocument Doc{"my-transformation", SourceMgr, {}};
  json::Value V = writeAndParse(Doc);

  const json::Object *Root = V.getAsObject();
  ASSERT_NE(Root, nullptr);
  const json::Array *Runs = Root->getArray("runs");
  ASSERT_NE(Runs, nullptr);
  ASSERT_EQ(Runs->size(), 1u);
  const json::Object *Driver =
      (*Runs)[0].getAsObject()->getObject("tool")->getObject("driver");
  ASSERT_NE(Driver, nullptr);
  EXPECT_EQ(*Driver->getString("name"), "clang-ssaf");
  EXPECT_NE(Driver->getString("fullName")->find("my-transformation"),
            StringRef::npos);
  EXPECT_EQ(*Driver->getString("version"), CLANG_VERSION_STRING);
}

TEST_F(SARIFFormatTest, LevelMapping) {
  ReportDocument Doc{"t",
                     SourceMgr,
                     {
                         {"r-note", clang::SarifResultLevel::Note, {}, "n"},
                         {"r-warn", clang::SarifResultLevel::Warning, {}, "w"},
                         {"r-error", clang::SarifResultLevel::Error, {}, "e"},
                         {"r-none", clang::SarifResultLevel::None, {}, "x"},
                     }};
  json::Value V = writeAndParse(Doc);

  const json::Array *Results =
      V.getAsObject()->getArray("runs")->front().getAsObject()->getArray(
          "results");
  ASSERT_NE(Results, nullptr);
  ASSERT_EQ(Results->size(), 4u);
  EXPECT_EQ(*(*Results)[0].getAsObject()->getString("level"), "note");
  EXPECT_EQ(*(*Results)[1].getAsObject()->getString("level"), "warning");
  EXPECT_EQ(*(*Results)[2].getAsObject()->getString("level"), "error");
  EXPECT_EQ(*(*Results)[3].getAsObject()->getString("level"), "none");
}

TEST_F(SARIFFormatTest, InvalidRangeOmitsLocations) {
  ReportDocument Doc{
      "t",
      SourceMgr,
      {{"r", clang::SarifResultLevel::Note, clang::CharSourceRange{}, "msg"}}};
  json::Value V = writeAndParse(Doc);
  const json::Object *Result = V.getAsObject()
                                   ->getArray("runs")
                                   ->front()
                                   .getAsObject()
                                   ->getArray("results")
                                   ->front()
                                   .getAsObject();
  EXPECT_EQ(Result->get("locations"), nullptr);
}

TEST_F(SARIFFormatTest, EmptyResultsValidDocument) {
  ReportDocument Doc{"t", SourceMgr, {}};
  json::Value V = writeAndParse(Doc);
  const json::Object *Run =
      V.getAsObject()->getArray("runs")->front().getAsObject();
  // SARIF allows the results array to be absent or empty for an empty run.
  if (const json::Array *Results = Run->getArray("results"))
    EXPECT_TRUE(Results->empty());
}

TEST_F(SARIFFormatTest, NoFixKey) {
  ReportDocument Doc{
      "t", SourceMgr, {{"r", clang::SarifResultLevel::Warning, {}, "msg"}}};
  json::Value V = writeAndParse(Doc);
  const json::Object *Result = V.getAsObject()
                                   ->getArray("runs")
                                   ->front()
                                   .getAsObject()
                                   ->getArray("results")
                                   ->front()
                                   .getAsObject();
  EXPECT_EQ(Result->get("fix"), nullptr);
  EXPECT_EQ(Result->get("fixes"), nullptr);
}

TEST_F(SARIFFormatTest, EmptyRuleIdAccepted) {
  ReportDocument Doc{
      "t", SourceMgr, {{"", clang::SarifResultLevel::Note, {}, "m"}}};
  json::Value V = writeAndParse(Doc);
  const json::Object *Result = V.getAsObject()
                                   ->getArray("runs")
                                   ->front()
                                   .getAsObject()
                                   ->getArray("results")
                                   ->front()
                                   .getAsObject();
  ASSERT_NE(Result->getString("ruleId"), std::nullopt);
  EXPECT_EQ(*Result->getString("ruleId"), "");
}

TEST_F(SARIFFormatTest, DistinctRuleIdsDeduplicated) {
  ReportDocument Doc{"t",
                     SourceMgr,
                     {
                         {"a", clang::SarifResultLevel::Note, {}, "m1"},
                         {"b", clang::SarifResultLevel::Note, {}, "m2"},
                         {"a", clang::SarifResultLevel::Note, {}, "m3"},
                     }};
  json::Value V = writeAndParse(Doc);
  const json::Array *Rules = V.getAsObject()
                                 ->getArray("runs")
                                 ->front()
                                 .getAsObject()
                                 ->getObject("tool")
                                 ->getObject("driver")
                                 ->getArray("rules");
  ASSERT_NE(Rules, nullptr);
  EXPECT_EQ(Rules->size(), 2u);
}

} // namespace
