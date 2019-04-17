//===-- HeadersTests.cpp - Include headers unit tests -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Headers.h"

#include "Compiler.h"
#include "TestFS.h"
#include "TestTU.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

class HeadersTest : public ::testing::Test {
public:
  HeadersTest() {
    CDB.ExtraClangFlags = {SearchDirArg.c_str()};
    FS.Files[MainFile] = "";
    // Make sure directory sub/ exists.
    FS.Files[testPath("sub/EMPTY")] = "";
  }

private:
  std::unique_ptr<CompilerInstance> setupClang() {
    auto Cmd = CDB.getCompileCommand(MainFile);
    assert(static_cast<bool>(Cmd));
    auto VFS = FS.getFileSystem();
    VFS->setCurrentWorkingDirectory(Cmd->Directory);

    ParseInputs PI;
    PI.CompileCommand = *Cmd;
    PI.FS = VFS;
    auto CI = buildCompilerInvocation(PI);
    EXPECT_TRUE(static_cast<bool>(CI));
    // The diagnostic options must be set before creating a CompilerInstance.
    CI->getDiagnosticOpts().IgnoreWarnings = true;
    auto Clang = prepareCompilerInstance(
        std::move(CI), /*Preamble=*/nullptr,
        llvm::MemoryBuffer::getMemBuffer(FS.Files[MainFile], MainFile), VFS,
        IgnoreDiags);

    EXPECT_FALSE(Clang->getFrontendOpts().Inputs.empty());
    return Clang;
  }

protected:
  IncludeStructure collectIncludes() {
    auto Clang = setupClang();
    PreprocessOnlyAction Action;
    EXPECT_TRUE(
        Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0]));
    IncludeStructure Includes;
    Clang->getPreprocessor().addPPCallbacks(
        collectIncludeStructureCallback(Clang->getSourceManager(), &Includes));
    EXPECT_TRUE(Action.Execute());
    Action.EndSourceFile();
    return Includes;
  }

  // Calculates the include path, or returns "" on error or header should not be
  // inserted.
  std::string calculate(PathRef Original, PathRef Preferred = "",
                        const std::vector<Inclusion> &Inclusions = {}) {
    auto Clang = setupClang();
    PreprocessOnlyAction Action;
    EXPECT_TRUE(
        Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0]));

    if (Preferred.empty())
      Preferred = Original;
    auto ToHeaderFile = [](llvm::StringRef Header) {
      return HeaderFile{Header,
                        /*Verbatim=*/!llvm::sys::path::is_absolute(Header)};
    };

    IncludeInserter Inserter(MainFile, /*Code=*/"", format::getLLVMStyle(),
                             CDB.getCompileCommand(MainFile)->Directory,
                             &Clang->getPreprocessor().getHeaderSearchInfo());
    for (const auto &Inc : Inclusions)
      Inserter.addExisting(Inc);
    auto Inserted = ToHeaderFile(Preferred);
    if (!Inserter.shouldInsertInclude(Original, Inserted))
      return "";
    std::string Path = Inserter.calculateIncludePath(Inserted);
    Action.EndSourceFile();
    return Path;
  }

  llvm::Optional<TextEdit> insert(llvm::StringRef VerbatimHeader) {
    auto Clang = setupClang();
    PreprocessOnlyAction Action;
    EXPECT_TRUE(
        Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0]));

    IncludeInserter Inserter(MainFile, /*Code=*/"", format::getLLVMStyle(),
                             CDB.getCompileCommand(MainFile)->Directory,
                             &Clang->getPreprocessor().getHeaderSearchInfo());
    auto Edit = Inserter.insert(VerbatimHeader);
    Action.EndSourceFile();
    return Edit;
  }

  MockFSProvider FS;
  MockCompilationDatabase CDB;
  std::string MainFile = testPath("main.cpp");
  std::string Subdir = testPath("sub");
  std::string SearchDirArg = (llvm::Twine("-I") + Subdir).str();
  IgnoringDiagConsumer IgnoreDiags;
};

MATCHER_P(Written, Name, "") { return arg.Written == Name; }
MATCHER_P(Resolved, Name, "") { return arg.Resolved == Name; }
MATCHER_P(IncludeLine, N, "") { return arg.R.start.line == N; }

MATCHER_P2(Distance, File, D, "") {
  if (arg.getKey() != File)
    *result_listener << "file =" << arg.getKey().str();
  if (arg.getValue() != D)
    *result_listener << "distance =" << arg.getValue();
  return arg.getKey() == File && arg.getValue() == D;
}

TEST_F(HeadersTest, CollectRewrittenAndResolved) {
  FS.Files[MainFile] = R"cpp(
#include "sub/bar.h" // not shortest
)cpp";
  std::string BarHeader = testPath("sub/bar.h");
  FS.Files[BarHeader] = "";

  EXPECT_THAT(collectIncludes().MainFileIncludes,
              UnorderedElementsAre(
                  AllOf(Written("\"sub/bar.h\""), Resolved(BarHeader))));
  EXPECT_THAT(collectIncludes().includeDepth(MainFile),
              UnorderedElementsAre(Distance(MainFile, 0u),
                                   Distance(testPath("sub/bar.h"), 1u)));
}

TEST_F(HeadersTest, OnlyCollectInclusionsInMain) {
  std::string BazHeader = testPath("sub/baz.h");
  FS.Files[BazHeader] = "";
  std::string BarHeader = testPath("sub/bar.h");
  FS.Files[BarHeader] = R"cpp(
#include "baz.h"
)cpp";
  FS.Files[MainFile] = R"cpp(
#include "bar.h"
)cpp";
  EXPECT_THAT(
      collectIncludes().MainFileIncludes,
      UnorderedElementsAre(AllOf(Written("\"bar.h\""), Resolved(BarHeader))));
  EXPECT_THAT(collectIncludes().includeDepth(MainFile),
              UnorderedElementsAre(Distance(MainFile, 0u),
                                   Distance(testPath("sub/bar.h"), 1u),
                                   Distance(testPath("sub/baz.h"), 2u)));
  // includeDepth() also works for non-main files.
  EXPECT_THAT(collectIncludes().includeDepth(testPath("sub/bar.h")),
              UnorderedElementsAre(Distance(testPath("sub/bar.h"), 0u),
                                   Distance(testPath("sub/baz.h"), 1u)));
}

TEST_F(HeadersTest, PreambleIncludesPresentOnce) {
  // We use TestTU here, to ensure we use the preamble replay logic.
  // We're testing that the logic doesn't crash, and doesn't result in duplicate
  // includes. (We'd test more directly, but it's pretty well encapsulated!)
  auto TU = TestTU::withCode(R"cpp(
    #include "a.h"
    #include "a.h"
    void foo();
    #include "a.h"
  )cpp");
  TU.HeaderFilename = "a.h"; // suppress "not found".
  EXPECT_THAT(TU.build().getIncludeStructure().MainFileIncludes,
              ElementsAre(IncludeLine(1), IncludeLine(2), IncludeLine(4)));
}

TEST_F(HeadersTest, UnResolvedInclusion) {
  FS.Files[MainFile] = R"cpp(
#include "foo.h"
)cpp";

  EXPECT_THAT(collectIncludes().MainFileIncludes,
              UnorderedElementsAre(AllOf(Written("\"foo.h\""), Resolved(""))));
  EXPECT_THAT(collectIncludes().includeDepth(MainFile),
              UnorderedElementsAre(Distance(MainFile, 0u)));
}

TEST_F(HeadersTest, InsertInclude) {
  std::string Path = testPath("sub/bar.h");
  FS.Files[Path] = "";
  EXPECT_EQ(calculate(Path), "\"bar.h\"");
}

TEST_F(HeadersTest, DoNotInsertIfInSameFile) {
  MainFile = testPath("main.h");
  EXPECT_EQ(calculate(MainFile), "");
}

TEST_F(HeadersTest, ShortenedInclude) {
  std::string BarHeader = testPath("sub/bar.h");
  EXPECT_EQ(calculate(BarHeader), "\"bar.h\"");
}

TEST_F(HeadersTest, NotShortenedInclude) {
  std::string BarHeader = testPath("sub-2/bar.h");
  EXPECT_EQ(calculate(BarHeader, ""), "\"" + BarHeader + "\"");
}

TEST_F(HeadersTest, PreferredHeader) {
  std::string BarHeader = testPath("sub/bar.h");
  EXPECT_EQ(calculate(BarHeader, "<bar>"), "<bar>");

  std::string BazHeader = testPath("sub/baz.h");
  EXPECT_EQ(calculate(BarHeader, BazHeader), "\"baz.h\"");
}

TEST_F(HeadersTest, DontInsertDuplicatePreferred) {
  Inclusion Inc;
  Inc.Written = "\"bar.h\"";
  Inc.Resolved = "";
  EXPECT_EQ(calculate(testPath("sub/bar.h"), "\"bar.h\"", {Inc}), "");
  EXPECT_EQ(calculate("\"x.h\"", "\"bar.h\"", {Inc}), "");
}

TEST_F(HeadersTest, DontInsertDuplicateResolved) {
  Inclusion Inc;
  Inc.Written = "fake-bar.h";
  Inc.Resolved = testPath("sub/bar.h");
  EXPECT_EQ(calculate(Inc.Resolved, "", {Inc}), "");
  // Do not insert preferred.
  EXPECT_EQ(calculate(Inc.Resolved, "\"BAR.h\"", {Inc}), "");
}

TEST_F(HeadersTest, PreferInserted) {
  auto Edit = insert("<y>");
  EXPECT_TRUE(Edit.hasValue());
  EXPECT_TRUE(StringRef(Edit->newText).contains("<y>"));
}

TEST(Headers, NoHeaderSearchInfo) {
  std::string MainFile = testPath("main.cpp");
  IncludeInserter Inserter(MainFile, /*Code=*/"", format::getLLVMStyle(),
                           /*BuildDir=*/"", /*HeaderSearchInfo=*/nullptr);

  auto HeaderPath = testPath("sub/bar.h");
  auto Inserting = HeaderFile{HeaderPath, /*Verbatim=*/false};
  auto Verbatim = HeaderFile{"<x>", /*Verbatim=*/true};

  EXPECT_EQ(Inserter.calculateIncludePath(Inserting),
            "\"" + HeaderPath + "\"");
  EXPECT_EQ(Inserter.shouldInsertInclude(HeaderPath, Inserting), false);

  EXPECT_EQ(Inserter.calculateIncludePath(Verbatim), "<x>");
  EXPECT_EQ(Inserter.shouldInsertInclude(HeaderPath, Verbatim), true);
}

} // namespace
} // namespace clangd
} // namespace clang
