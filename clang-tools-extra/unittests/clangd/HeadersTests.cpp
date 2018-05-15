//===-- HeadersTests.cpp - Include headers unit tests -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Headers.h"

#include "Compiler.h"
#include "TestFS.h"
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

    std::vector<const char *> Argv;
    for (const auto &S : Cmd->CommandLine)
      Argv.push_back(S.c_str());
    auto CI = clang::createInvocationFromCommandLine(
        Argv,
        CompilerInstance::createDiagnostics(new DiagnosticOptions(),
                                            &IgnoreDiags, false),
        VFS);
    EXPECT_TRUE(static_cast<bool>(CI));
    CI->getFrontendOpts().DisableFree = false;

    // The diagnostic options must be set before creating a CompilerInstance.
    CI->getDiagnosticOpts().IgnoreWarnings = true;
    auto Clang = prepareCompilerInstance(
        std::move(CI), /*Preamble=*/nullptr,
        llvm::MemoryBuffer::getMemBuffer(FS.Files[MainFile], MainFile),
        std::make_shared<PCHContainerOperations>(), VFS, IgnoreDiags);

    EXPECT_FALSE(Clang->getFrontendOpts().Inputs.empty());
    return Clang;
  }

protected:
  std::vector<Inclusion> collectIncludes() {
    auto Clang = setupClang();
    PreprocessOnlyAction Action;
    EXPECT_TRUE(
        Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0]));
    std::vector<Inclusion> Inclusions;
    Clang->getPreprocessor().addPPCallbacks(collectInclusionsInMainFileCallback(
        Clang->getSourceManager(),
        [&](Inclusion Inc) { Inclusions.push_back(std::move(Inc)); }));
    EXPECT_TRUE(Action.Execute());
    Action.EndSourceFile();
    return Inclusions;
  }

  // Calculates the include path, or returns "" on error.
  std::string calculate(PathRef Original, PathRef Preferred = "",
                        const std::vector<Inclusion> &Inclusions = {},
                        bool ExpectError = false) {
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

    auto Path = calculateIncludePath(
        MainFile, CDB.getCompileCommand(MainFile)->Directory,
        Clang->getPreprocessor().getHeaderSearchInfo(), Inclusions,
        ToHeaderFile(Original), ToHeaderFile(Preferred));
    Action.EndSourceFile();
    if (!Path) {
      llvm::consumeError(Path.takeError());
      EXPECT_TRUE(ExpectError);
      return std::string();
    } else {
      EXPECT_FALSE(ExpectError);
    }
    return std::move(*Path);
  }

  Expected<Optional<TextEdit>>
  insert(const HeaderFile &DeclaringHeader, const HeaderFile &InsertedHeader,
         const std::vector<Inclusion> &ExistingInclusions = {}) {
    auto Clang = setupClang();
    PreprocessOnlyAction Action;
    EXPECT_TRUE(
        Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0]));

    IncludeInserter Inserter(MainFile, /*Code=*/"", format::getLLVMStyle(),
                             CDB.getCompileCommand(MainFile)->Directory,
                             Clang->getPreprocessor().getHeaderSearchInfo());
    for (const auto &Inc : ExistingInclusions)
      Inserter.addExisting(Inc);

    auto Edit = Inserter.insert(DeclaringHeader, InsertedHeader);
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

TEST_F(HeadersTest, CollectRewrittenAndResolved) {
  FS.Files[MainFile] = R"cpp(
#include "sub/bar.h" // not shortest
)cpp";
  std::string BarHeader = testPath("sub/bar.h");
  FS.Files[BarHeader] = "";

  EXPECT_THAT(collectIncludes(),
              UnorderedElementsAre(
                  AllOf(Written("\"sub/bar.h\""), Resolved(BarHeader))));
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
      collectIncludes(),
      UnorderedElementsAre(AllOf(Written("\"bar.h\""), Resolved(BarHeader))));
}

TEST_F(HeadersTest, UnResolvedInclusion) {
  FS.Files[MainFile] = R"cpp(
#include "foo.h"
)cpp";

  EXPECT_THAT(collectIncludes(),
              UnorderedElementsAre(AllOf(Written("\"foo.h\""), Resolved(""))));
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
  std::vector<Inclusion> Inclusions = {
      {Range(), /*Written*/ "\"bar.h\"", /*Resolved*/ ""}};
  EXPECT_EQ(calculate(testPath("sub/bar.h"), "\"bar.h\"", Inclusions), "");
  EXPECT_EQ(calculate("\"x.h\"", "\"bar.h\"", Inclusions), "");
}

TEST_F(HeadersTest, DontInsertDuplicateResolved) {
  std::string BarHeader = testPath("sub/bar.h");
  std::vector<Inclusion> Inclusions = {
      {Range(), /*Written*/ "fake-bar.h", /*Resolved*/ BarHeader}};
  EXPECT_EQ(calculate(BarHeader, "", Inclusions), "");
  // Do not insert preferred.
  EXPECT_EQ(calculate(BarHeader, "\"BAR.h\"", Inclusions), "");
}

HeaderFile literal(StringRef Include) {
  HeaderFile H{Include, /*Verbatim=*/true};
  assert(H.valid());
  return H;
}

TEST_F(HeadersTest, PreferInserted) {
  auto Edit = insert(literal("<x>"), literal("<y>"));
  EXPECT_TRUE(static_cast<bool>(Edit));
  EXPECT_TRUE(llvm::StringRef((*Edit)->newText).contains("<y>"));
}

TEST_F(HeadersTest, ExistingInclusion) {
  Inclusion Existing{Range(), /*Written=*/"<c.h>", /*Resolved=*/""};
  auto Edit = insert(literal("<c.h>"), literal("<c.h>"), {Existing});
  EXPECT_TRUE(static_cast<bool>(Edit));
  EXPECT_FALSE(static_cast<bool>(*Edit));
}

TEST_F(HeadersTest, ValidateHeaders) {
  HeaderFile InvalidHeader{"a.h", /*Verbatim=*/true};
  assert(!InvalidHeader.valid());
  auto Edit = insert(InvalidHeader, literal("\"c.h\""));
  EXPECT_FALSE(static_cast<bool>(Edit));
  llvm::consumeError(Edit.takeError());
}

} // namespace
} // namespace clangd
} // namespace clang
