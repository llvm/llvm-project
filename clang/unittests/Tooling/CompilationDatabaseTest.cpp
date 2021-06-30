//===- unittest/Tooling/CompilationDatabaseTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/FileMatchTrie.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

using testing::ElementsAre;
using testing::EndsWith;

static void expectFailure(StringRef JSONDatabase, StringRef Explanation) {
  std::string ErrorMessage;
  EXPECT_EQ(nullptr,
            JSONCompilationDatabase::loadFromBuffer(JSONDatabase, ErrorMessage,
                                                    JSONCommandLineSyntax::Gnu))
      << "Expected an error because of: " << Explanation.str();
}

TEST(JSONCompilationDatabase, ErrsOnInvalidFormat) {
  expectFailure("", "Empty database");
  expectFailure("{", "Invalid JSON");
  expectFailure("[[]]", "Array instead of object");
  expectFailure("[{\"a\":[]}]", "Array instead of value");
  expectFailure("[{\"a\":\"b\"}]", "Unknown key");
  expectFailure("[{[]:\"\"}]", "Incorrectly typed entry");
  expectFailure("[{}]", "Empty entry");
  expectFailure("[{\"directory\":\"\",\"command\":\"\"}]", "Missing file");
  expectFailure("[{\"directory\":\"\",\"file\":\"\"}]", "Missing command or arguments");
  expectFailure("[{\"command\":\"\",\"file\":\"\"}]", "Missing directory");
  expectFailure("[{\"directory\":\"\",\"arguments\":[]}]", "Missing file");
  expectFailure("[{\"arguments\":\"\",\"file\":\"\"}]", "Missing directory");
  expectFailure("[{\"directory\":\"\",\"arguments\":\"\",\"file\":\"\"}]", "Arguments not array");
  expectFailure("[{\"directory\":\"\",\"command\":[],\"file\":\"\"}]", "Command not string");
  expectFailure("[{\"directory\":\"\",\"arguments\":[[]],\"file\":\"\"}]",
                "Arguments contain non-string");
  expectFailure("[{\"output\":[]}]", "Expected strings as value.");
}

static std::vector<std::string> getAllFiles(StringRef JSONDatabase,
                                            std::string &ErrorMessage,
                                            JSONCommandLineSyntax Syntax) {
  std::unique_ptr<CompilationDatabase> Database(
      JSONCompilationDatabase::loadFromBuffer(JSONDatabase, ErrorMessage,
                                              Syntax));
  if (!Database) {
    ADD_FAILURE() << ErrorMessage;
    return std::vector<std::string>();
  }
  return Database->getAllFiles();
}

static std::vector<CompileCommand>
getAllCompileCommands(JSONCommandLineSyntax Syntax, StringRef JSONDatabase,
                      std::string &ErrorMessage) {
  std::unique_ptr<CompilationDatabase> Database(
      JSONCompilationDatabase::loadFromBuffer(JSONDatabase, ErrorMessage,
                                              Syntax));
  if (!Database) {
    ADD_FAILURE() << ErrorMessage;
    return std::vector<CompileCommand>();
  }
  return Database->getAllCompileCommands();
}

TEST(JSONCompilationDatabase, GetAllFiles) {
  std::string ErrorMessage;
  EXPECT_EQ(std::vector<std::string>(),
            getAllFiles("[]", ErrorMessage, JSONCommandLineSyntax::Gnu))
      << ErrorMessage;

  std::vector<std::string> expected_files;
  SmallString<16> PathStorage;
  llvm::sys::path::native("//net/dir/file1", PathStorage);
  expected_files.push_back(std::string(PathStorage.str()));
  llvm::sys::path::native("//net/dir/file2", PathStorage);
  expected_files.push_back(std::string(PathStorage.str()));
  llvm::sys::path::native("//net/file1", PathStorage);
  expected_files.push_back(std::string(PathStorage.str()));
  EXPECT_EQ(expected_files,
            getAllFiles("[{\"directory\":\"//net/dir\","
                        "\"command\":\"command\","
                        "\"file\":\"file1\"},"
                        " {\"directory\":\"//net/dir\","
                        "\"command\":\"command\","
                        "\"file\":\"../file1\"},"
                        " {\"directory\":\"//net/dir\","
                        "\"command\":\"command\","
                        "\"file\":\"file2\"}]",
                        ErrorMessage, JSONCommandLineSyntax::Gnu))
      << ErrorMessage;
}

TEST(JSONCompilationDatabase, GetAllCompileCommands) {
  std::string ErrorMessage;
  EXPECT_EQ(
      0u, getAllCompileCommands(JSONCommandLineSyntax::Gnu, "[]", ErrorMessage)
              .size())
      << ErrorMessage;

  StringRef Directory1("//net/dir1");
  StringRef FileName1("file1");
  StringRef Command1("command1");
  StringRef Output1("file1.o");
  StringRef Directory2("//net/dir2");
  StringRef FileName2("file2");
  StringRef Command2("command2");
  StringRef Output2("");

  std::vector<CompileCommand> Commands = getAllCompileCommands(
      JSONCommandLineSyntax::Gnu,
      ("[{\"directory\":\"" + Directory1 + "\"," + "\"command\":\"" + Command1 +
       "\","
       "\"file\":\"" +
       FileName1 + "\", \"output\":\"" +
       Output1 + "\"},"
                   " {\"directory\":\"" +
       Directory2 + "\"," + "\"command\":\"" + Command2 + "\","
                                                          "\"file\":\"" +
       FileName2 + "\"}]")
          .str(),
      ErrorMessage);
  EXPECT_EQ(2U, Commands.size()) << ErrorMessage;
  EXPECT_EQ(Directory1, Commands[0].Directory) << ErrorMessage;
  EXPECT_EQ(FileName1, Commands[0].Filename) << ErrorMessage;
  EXPECT_EQ(Output1, Commands[0].Output) << ErrorMessage;
  ASSERT_EQ(1u, Commands[0].CommandLine.size());
  EXPECT_EQ(Command1, Commands[0].CommandLine[0]) << ErrorMessage;
  EXPECT_EQ(Directory2, Commands[1].Directory) << ErrorMessage;
  EXPECT_EQ(FileName2, Commands[1].Filename) << ErrorMessage;
  EXPECT_EQ(Output2, Commands[1].Output) << ErrorMessage;
  ASSERT_EQ(1u, Commands[1].CommandLine.size());
  EXPECT_EQ(Command2, Commands[1].CommandLine[0]) << ErrorMessage;

  // Check that order is preserved.
  Commands = getAllCompileCommands(
      JSONCommandLineSyntax::Gnu,
      ("[{\"directory\":\"" + Directory2 + "\"," + "\"command\":\"" + Command2 +
       "\","
       "\"file\":\"" +
       FileName2 + "\"},"
                   " {\"directory\":\"" +
       Directory1 + "\"," + "\"command\":\"" + Command1 + "\","
                                                          "\"file\":\"" +
       FileName1 + "\"}]")
          .str(),
      ErrorMessage);
  EXPECT_EQ(2U, Commands.size()) << ErrorMessage;
  EXPECT_EQ(Directory2, Commands[0].Directory) << ErrorMessage;
  EXPECT_EQ(FileName2, Commands[0].Filename) << ErrorMessage;
  ASSERT_EQ(1u, Commands[0].CommandLine.size());
  EXPECT_EQ(Command2, Commands[0].CommandLine[0]) << ErrorMessage;
  EXPECT_EQ(Directory1, Commands[1].Directory) << ErrorMessage;
  EXPECT_EQ(FileName1, Commands[1].Filename) << ErrorMessage;
  ASSERT_EQ(1u, Commands[1].CommandLine.size());
  EXPECT_EQ(Command1, Commands[1].CommandLine[0]) << ErrorMessage;
}

static CompileCommand findCompileArgsInJsonDatabase(StringRef FileName,
                                                    std::string JSONDatabase,
                                                    std::string &ErrorMessage) {
  std::unique_ptr<CompilationDatabase> Database(
      JSONCompilationDatabase::loadFromBuffer(JSONDatabase, ErrorMessage,
                                              JSONCommandLineSyntax::Gnu));
  if (!Database)
    return CompileCommand();
  // Overwrite the string to verify we're not reading from it later.
  JSONDatabase.assign(JSONDatabase.size(), '*');
  std::vector<CompileCommand> Commands = Database->getCompileCommands(FileName);
  EXPECT_LE(Commands.size(), 1u);
  if (Commands.empty())
    return CompileCommand();
  return Commands[0];
}

TEST(JSONCompilationDatabase, ArgumentsPreferredOverCommand) {
   StringRef Directory("//net/dir");
   StringRef FileName("//net/dir/filename");
   StringRef Command("command");
   StringRef Arguments = "arguments";
   Twine ArgumentsAccumulate;
   std::string ErrorMessage;
   CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
      FileName,
      ("[{\"directory\":\"" + Directory + "\","
         "\"arguments\":[\"" + Arguments + "\"],"
         "\"command\":\"" + Command + "\","
         "\"file\":\"" + FileName + "\"}]").str(),
      ErrorMessage);
   EXPECT_EQ(Directory, FoundCommand.Directory) << ErrorMessage;
   EXPECT_EQ(1u, FoundCommand.CommandLine.size()) << ErrorMessage;
   EXPECT_EQ(Arguments, FoundCommand.CommandLine[0]) << ErrorMessage;
}

struct FakeComparator : public PathComparator {
  ~FakeComparator() override {}
  bool equivalent(StringRef FileA, StringRef FileB) const override {
    return FileA.equals_insensitive(FileB);
  }
};

class FileMatchTrieTest : public ::testing::Test {
protected:
  FileMatchTrieTest() : Trie(new FakeComparator()) {}

  StringRef find(StringRef Path) {
    llvm::raw_string_ostream ES(Error);
    return Trie.findEquivalent(Path, ES);
  }

  FileMatchTrie Trie;
  std::string Error;
};

TEST_F(FileMatchTrieTest, InsertingRelativePath) {
  Trie.insert("//net/path/file.cc");
  Trie.insert("file.cc");
  EXPECT_EQ("//net/path/file.cc", find("//net/path/file.cc"));
}

TEST_F(FileMatchTrieTest, MatchingRelativePath) {
  EXPECT_EQ("", find("file.cc"));
}

TEST_F(FileMatchTrieTest, ReturnsBestResults) {
  Trie.insert("//net/d/c/b.cc");
  Trie.insert("//net/d/b/b.cc");
  EXPECT_EQ("//net/d/b/b.cc", find("//net/d/b/b.cc"));
}

TEST_F(FileMatchTrieTest, HandlesSymlinks) {
  Trie.insert("//net/AA/file.cc");
  EXPECT_EQ("//net/AA/file.cc", find("//net/aa/file.cc"));
}

TEST_F(FileMatchTrieTest, ReportsSymlinkAmbiguity) {
  Trie.insert("//net/Aa/file.cc");
  Trie.insert("//net/aA/file.cc");
  EXPECT_TRUE(find("//net/aa/file.cc").empty());
  EXPECT_EQ("Path is ambiguous", Error);
}

TEST_F(FileMatchTrieTest, LongerMatchingSuffixPreferred) {
  Trie.insert("//net/src/Aa/file.cc");
  Trie.insert("//net/src/aA/file.cc");
  Trie.insert("//net/SRC/aa/file.cc");
  EXPECT_EQ("//net/SRC/aa/file.cc", find("//net/src/aa/file.cc"));
}

TEST_F(FileMatchTrieTest, EmptyTrie) {
  EXPECT_TRUE(find("//net/some/path").empty());
}

TEST_F(FileMatchTrieTest, NoResult) {
  Trie.insert("//net/somepath/otherfile.cc");
  Trie.insert("//net/otherpath/somefile.cc");
  EXPECT_EQ("", find("//net/somepath/somefile.cc"));
}

TEST_F(FileMatchTrieTest, RootElementDifferent) {
  Trie.insert("//net/path/file.cc");
  Trie.insert("//net/otherpath/file.cc");
  EXPECT_EQ("//net/path/file.cc", find("//net/path/file.cc"));
}

TEST_F(FileMatchTrieTest, CannotResolveRelativePath) {
  EXPECT_EQ("", find("relative-path.cc"));
  EXPECT_EQ("Cannot resolve relative paths", Error);
}

TEST_F(FileMatchTrieTest, SingleFile) {
  Trie.insert("/root/RootFile.cc");
  EXPECT_EQ("", find("/root/rootfile.cc"));
  // Add subpath to avoid `if (Children.empty())` special case
  // which we hit at previous `find()`.
  Trie.insert("/root/otherpath/OtherFile.cc");
  EXPECT_EQ("", find("/root/rootfile.cc"));
}

TEST(findCompileArgsInJsonDatabase, FindsNothingIfEmpty) {
  std::string ErrorMessage;
  CompileCommand NotFound = findCompileArgsInJsonDatabase(
    "a-file.cpp", "", ErrorMessage);
  EXPECT_TRUE(NotFound.CommandLine.empty()) << ErrorMessage;
  EXPECT_TRUE(NotFound.Directory.empty()) << ErrorMessage;
}

TEST(findCompileArgsInJsonDatabase, ReadsSingleEntry) {
  StringRef Directory("//net/some/directory");
  StringRef FileName("//net/path/to/a-file.cpp");
  StringRef Command("//net/path/to/compiler and some arguments");
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    FileName,
    ("[{\"directory\":\"" + Directory + "\"," +
       "\"command\":\"" + Command + "\","
       "\"file\":\"" + FileName + "\"}]").str(),
    ErrorMessage);
  EXPECT_EQ(Directory, FoundCommand.Directory) << ErrorMessage;
  ASSERT_EQ(4u, FoundCommand.CommandLine.size()) << ErrorMessage;
  EXPECT_EQ("//net/path/to/compiler",
            FoundCommand.CommandLine[0]) << ErrorMessage;
  EXPECT_EQ("and", FoundCommand.CommandLine[1]) << ErrorMessage;
  EXPECT_EQ("some", FoundCommand.CommandLine[2]) << ErrorMessage;
  EXPECT_EQ("arguments", FoundCommand.CommandLine[3]) << ErrorMessage;

  CompileCommand NotFound = findCompileArgsInJsonDatabase(
    "a-file.cpp",
    ("[{\"directory\":\"" + Directory + "\"," +
       "\"command\":\"" + Command + "\","
       "\"file\":\"" + FileName + "\"}]").str(),
    ErrorMessage);
  EXPECT_TRUE(NotFound.Directory.empty()) << ErrorMessage;
  EXPECT_TRUE(NotFound.CommandLine.empty()) << ErrorMessage;
}

TEST(findCompileArgsInJsonDatabase, ReadsCompileCommandLinesWithSpaces) {
  StringRef Directory("//net/some/directory");
  StringRef FileName("//net/path/to/a-file.cpp");
  StringRef Command("\\\"//net/path to compiler\\\" \\\"and an argument\\\"");
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    FileName,
    ("[{\"directory\":\"" + Directory + "\"," +
       "\"command\":\"" + Command + "\","
       "\"file\":\"" + FileName + "\"}]").str(),
    ErrorMessage);
  ASSERT_EQ(2u, FoundCommand.CommandLine.size());
  EXPECT_EQ("//net/path to compiler",
            FoundCommand.CommandLine[0]) << ErrorMessage;
  EXPECT_EQ("and an argument", FoundCommand.CommandLine[1]) << ErrorMessage;
}

TEST(findCompileArgsInJsonDatabase, ReadsDirectoryWithSpaces) {
  StringRef Directory("//net/some directory / with spaces");
  StringRef FileName("//net/path/to/a-file.cpp");
  StringRef Command("a command");
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    FileName,
    ("[{\"directory\":\"" + Directory + "\"," +
       "\"command\":\"" + Command + "\","
       "\"file\":\"" + FileName + "\"}]").str(),
    ErrorMessage);
  EXPECT_EQ(Directory, FoundCommand.Directory) << ErrorMessage;
}

TEST(findCompileArgsInJsonDatabase, FindsEntry) {
  StringRef Directory("//net/directory");
  StringRef FileName("file");
  StringRef Command("command");
  std::string JsonDatabase = "[";
  for (int I = 0; I < 10; ++I) {
    if (I > 0) JsonDatabase += ",";
    JsonDatabase +=
      ("{\"directory\":\"" + Directory + Twine(I) + "\"," +
        "\"command\":\"" + Command + Twine(I) + "\","
        "\"file\":\"" + FileName + Twine(I) + "\"}").str();
  }
  JsonDatabase += "]";
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    "//net/directory4/file4", JsonDatabase, ErrorMessage);
  EXPECT_EQ("//net/directory4", FoundCommand.Directory) << ErrorMessage;
  ASSERT_EQ(1u, FoundCommand.CommandLine.size()) << ErrorMessage;
  EXPECT_EQ("command4", FoundCommand.CommandLine[0]) << ErrorMessage;
}

TEST(findCompileArgsInJsonDatabase, ParsesCompilerWrappers) {
  std::vector<std::pair<std::string, std::string>> Cases = {
      {"distcc gcc foo.c", "gcc foo.c"},
      {"gomacc clang++ foo.c", "clang++ foo.c"},
      {"sccache clang++ foo.c", "clang++ foo.c"},
      {"ccache gcc foo.c", "gcc foo.c"},
      {"ccache.exe gcc foo.c", "gcc foo.c"},
      {"ccache g++.exe foo.c", "g++.exe foo.c"},
      {"ccache distcc gcc foo.c", "gcc foo.c"},

      {"distcc foo.c", "distcc foo.c"},
      {"distcc -I/foo/bar foo.c", "distcc -I/foo/bar foo.c"},
  };
  std::string ErrorMessage;

  for (const auto &Case : Cases) {
    std::string DB =
        R"([{"directory":"//net/dir", "file":"//net/dir/foo.c", "command":")" +
        Case.first + "\"}]";
    CompileCommand FoundCommand =
        findCompileArgsInJsonDatabase("//net/dir/foo.c", DB, ErrorMessage);
    EXPECT_EQ(Case.second, llvm::join(FoundCommand.CommandLine, " "))
        << Case.first;
  }
}

static std::vector<std::string> unescapeJsonCommandLine(StringRef Command) {
  std::string JsonDatabase =
    ("[{\"directory\":\"//net/root\", \"file\":\"test\", \"command\": \"" +
     Command + "\"}]").str();
  std::string ErrorMessage;
  CompileCommand FoundCommand = findCompileArgsInJsonDatabase(
    "//net/root/test", JsonDatabase, ErrorMessage);
  EXPECT_TRUE(ErrorMessage.empty()) << ErrorMessage;
  return FoundCommand.CommandLine;
}

TEST(unescapeJsonCommandLine, ReturnsEmptyArrayOnEmptyString) {
  std::vector<std::string> Result = unescapeJsonCommandLine("");
  EXPECT_TRUE(Result.empty());
}

TEST(unescapeJsonCommandLine, SplitsOnSpaces) {
  std::vector<std::string> Result = unescapeJsonCommandLine("a b c");
  ASSERT_EQ(3ul, Result.size());
  EXPECT_EQ("a", Result[0]);
  EXPECT_EQ("b", Result[1]);
  EXPECT_EQ("c", Result[2]);
}

TEST(unescapeJsonCommandLine, MungesMultipleSpaces) {
  std::vector<std::string> Result = unescapeJsonCommandLine("   a   b   ");
  ASSERT_EQ(2ul, Result.size());
  EXPECT_EQ("a", Result[0]);
  EXPECT_EQ("b", Result[1]);
}

TEST(unescapeJsonCommandLine, UnescapesBackslashCharacters) {
  std::vector<std::string> Backslash = unescapeJsonCommandLine("a\\\\\\\\");
  ASSERT_EQ(1ul, Backslash.size());
  EXPECT_EQ("a\\", Backslash[0]);
  std::vector<std::string> Quote = unescapeJsonCommandLine("a\\\\\\\"");
  ASSERT_EQ(1ul, Quote.size());
  EXPECT_EQ("a\"", Quote[0]);
}

TEST(unescapeJsonCommandLine, DoesNotMungeSpacesBetweenQuotes) {
  std::vector<std::string> Result = unescapeJsonCommandLine("\\\"  a  b  \\\"");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_EQ("  a  b  ", Result[0]);
}

TEST(unescapeJsonCommandLine, AllowsMultipleQuotedArguments) {
  std::vector<std::string> Result = unescapeJsonCommandLine(
      "  \\\" a \\\"  \\\" b \\\"  ");
  ASSERT_EQ(2ul, Result.size());
  EXPECT_EQ(" a ", Result[0]);
  EXPECT_EQ(" b ", Result[1]);
}

TEST(unescapeJsonCommandLine, AllowsEmptyArgumentsInQuotes) {
  std::vector<std::string> Result = unescapeJsonCommandLine(
      "\\\"\\\"\\\"\\\"");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_TRUE(Result[0].empty()) << Result[0];
}

TEST(unescapeJsonCommandLine, ParsesEscapedQuotesInQuotedStrings) {
  std::vector<std::string> Result = unescapeJsonCommandLine(
      "\\\"\\\\\\\"\\\"");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_EQ("\"", Result[0]);
}

TEST(unescapeJsonCommandLine, ParsesMultipleArgumentsWithEscapedCharacters) {
  std::vector<std::string> Result = unescapeJsonCommandLine(
      "  \\\\\\\"  \\\"a \\\\\\\" b \\\"     \\\"and\\\\\\\\c\\\"   \\\\\\\"");
  ASSERT_EQ(4ul, Result.size());
  EXPECT_EQ("\"", Result[0]);
  EXPECT_EQ("a \" b ", Result[1]);
  EXPECT_EQ("and\\c", Result[2]);
  EXPECT_EQ("\"", Result[3]);
}

TEST(unescapeJsonCommandLine, ParsesStringsWithoutSpacesIntoSingleArgument) {
  std::vector<std::string> QuotedNoSpaces = unescapeJsonCommandLine(
      "\\\"a\\\"\\\"b\\\"");
  ASSERT_EQ(1ul, QuotedNoSpaces.size());
  EXPECT_EQ("ab", QuotedNoSpaces[0]);

  std::vector<std::string> MixedNoSpaces = unescapeJsonCommandLine(
      "\\\"a\\\"bcd\\\"ef\\\"\\\"\\\"\\\"g\\\"");
  ASSERT_EQ(1ul, MixedNoSpaces.size());
  EXPECT_EQ("abcdefg", MixedNoSpaces[0]);
}

TEST(unescapeJsonCommandLine, ParsesQuotedStringWithoutClosingQuote) {
  std::vector<std::string> Unclosed = unescapeJsonCommandLine("\\\"abc");
  ASSERT_EQ(1ul, Unclosed.size());
  EXPECT_EQ("abc", Unclosed[0]);

  std::vector<std::string> Empty = unescapeJsonCommandLine("\\\"");
  ASSERT_EQ(1ul, Empty.size());
  EXPECT_EQ("", Empty[0]);
}

TEST(unescapeJsonCommandLine, ParsesSingleQuotedString) {
  std::vector<std::string> Args = unescapeJsonCommandLine("a'\\\\b \\\"c\\\"'");
  ASSERT_EQ(1ul, Args.size());
  EXPECT_EQ("a\\b \"c\"", Args[0]);
}

TEST(FixedCompilationDatabase, ReturnsFixedCommandLine) {
  FixedCompilationDatabase Database(".", /*CommandLine*/ {"one", "two"});
  StringRef FileName("source");
  std::vector<CompileCommand> Result =
    Database.getCompileCommands(FileName);
  ASSERT_EQ(1ul, Result.size());
  EXPECT_EQ(".", Result[0].Directory);
  EXPECT_EQ(FileName, Result[0].Filename);
  EXPECT_THAT(Result[0].CommandLine,
              ElementsAre(EndsWith("clang-tool"), "one", "two", "source"));
}

TEST(FixedCompilationDatabase, GetAllFiles) {
  std::vector<std::string> CommandLine;
  CommandLine.push_back("one");
  CommandLine.push_back("two");
  FixedCompilationDatabase Database(".", CommandLine);

  EXPECT_EQ(0ul, Database.getAllFiles().size());
}

TEST(FixedCompilationDatabase, GetAllCompileCommands) {
  std::vector<std::string> CommandLine;
  CommandLine.push_back("one");
  CommandLine.push_back("two");
  FixedCompilationDatabase Database(".", CommandLine);

  EXPECT_EQ(0ul, Database.getAllCompileCommands().size());
}

TEST(FixedCompilationDatabase, FromBuffer) {
  const char *Data = R"(

 -DFOO=BAR

--baz

  )";
  std::string ErrorMsg;
  auto CDB =
      FixedCompilationDatabase::loadFromBuffer("/cdb/dir", Data, ErrorMsg);

  std::vector<CompileCommand> Result = CDB->getCompileCommands("/foo/bar.cc");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_EQ("/cdb/dir", Result.front().Directory);
  EXPECT_EQ("/foo/bar.cc", Result.front().Filename);
  EXPECT_THAT(
      Result.front().CommandLine,
      ElementsAre(EndsWith("clang-tool"), "-DFOO=BAR", "--baz", "/foo/bar.cc"));
}

TEST(ParseFixedCompilationDatabase, ReturnsNullOnEmptyArgumentList) {
  int Argc = 0;
  std::string ErrorMsg;
  std::unique_ptr<FixedCompilationDatabase> Database =
      FixedCompilationDatabase::loadFromCommandLine(Argc, nullptr, ErrorMsg);
  EXPECT_FALSE(Database);
  EXPECT_TRUE(ErrorMsg.empty());
  EXPECT_EQ(0, Argc);
}

TEST(ParseFixedCompilationDatabase, ReturnsNullWithoutDoubleDash) {
  int Argc = 2;
  const char *Argv[] = { "1", "2" };
  std::string ErrorMsg;
  std::unique_ptr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv, ErrorMsg));
  EXPECT_FALSE(Database);
  EXPECT_TRUE(ErrorMsg.empty());
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, ReturnsArgumentsAfterDoubleDash) {
  int Argc = 5;
  const char *Argv[] = {
    "1", "2", "--\0no-constant-folding", "-DDEF3", "-DDEF4"
  };
  std::string ErrorMsg;
  std::unique_ptr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv, ErrorMsg));
  ASSERT_TRUE((bool)Database);
  ASSERT_TRUE(ErrorMsg.empty());
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  ASSERT_THAT(Result[0].CommandLine, ElementsAre(EndsWith("clang-tool"),
                                                 "-DDEF3", "-DDEF4", "source"));
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, ReturnsEmptyCommandLine) {
  int Argc = 3;
  const char *Argv[] = { "1", "2", "--\0no-constant-folding" };
  std::string ErrorMsg;
  std::unique_ptr<FixedCompilationDatabase> Database =
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv, ErrorMsg);
  ASSERT_TRUE((bool)Database);
  ASSERT_TRUE(ErrorMsg.empty());
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  ASSERT_THAT(Result[0].CommandLine,
              ElementsAre(EndsWith("clang-tool"), "source"));
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, HandlesPositionalArgs) {
  const char *Argv[] = {"1", "2", "--", "-c", "somefile.cpp", "-DDEF3"};
  int Argc = sizeof(Argv) / sizeof(char*);
  std::string ErrorMsg;
  std::unique_ptr<FixedCompilationDatabase> Database =
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv, ErrorMsg);
  ASSERT_TRUE((bool)Database);
  ASSERT_TRUE(ErrorMsg.empty());
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  ASSERT_THAT(Result[0].CommandLine,
              ElementsAre(EndsWith("clang-tool"), "-c", "-DDEF3", "source"));
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, HandlesPositionalArgsSyntaxOnly) {
  // Adjust the given command line arguments to ensure that any positional
  // arguments in them are stripped.
  const char *Argv[] = {"--", "somefile.cpp", "-fsyntax-only", "-DDEF3"};
  int Argc = llvm::array_lengthof(Argv);
  std::string ErrorMessage;
  std::unique_ptr<CompilationDatabase> Database =
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv, ErrorMessage);
  ASSERT_TRUE((bool)Database);
  ASSERT_TRUE(ErrorMessage.empty());
  std::vector<CompileCommand> Result = Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  ASSERT_THAT(
      Result[0].CommandLine,
      ElementsAre(EndsWith("clang-tool"), "-fsyntax-only", "-DDEF3", "source"));
}

TEST(ParseFixedCompilationDatabase, HandlesArgv0) {
  const char *Argv[] = {"1", "2", "--", "mytool", "somefile.cpp"};
  int Argc = sizeof(Argv) / sizeof(char*);
  std::string ErrorMsg;
  std::unique_ptr<FixedCompilationDatabase> Database =
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv, ErrorMsg);
  ASSERT_TRUE((bool)Database);
  ASSERT_TRUE(ErrorMsg.empty());
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  std::vector<std::string> Expected;
  ASSERT_THAT(Result[0].CommandLine,
              ElementsAre(EndsWith("clang-tool"), "source"));
  EXPECT_EQ(2, Argc);
}

struct MemCDB : public CompilationDatabase {
  using EntryMap = llvm::StringMap<SmallVector<CompileCommand, 1>>;
  EntryMap Entries;
  MemCDB(const EntryMap &E) : Entries(E) {}

  std::vector<CompileCommand> getCompileCommands(StringRef F) const override {
    auto Ret = Entries.lookup(F);
    return {Ret.begin(), Ret.end()};
  }

  std::vector<std::string> getAllFiles() const override {
    std::vector<std::string> Result;
    for (const auto &Entry : Entries)
      Result.push_back(std::string(Entry.first()));
    return Result;
  }
};

class MemDBTest : public ::testing::Test {
protected:
  // Adds an entry to the underlying compilation database.
  // A flag is injected: -D <File>, so the command used can be identified.
  void add(StringRef File, StringRef Clang, StringRef Flags) {
    SmallVector<StringRef, 8> Argv = {Clang, File, "-D", File};
    llvm::SplitString(Flags, Argv);

    // Trim double quotation from the argumnets if any.
    for (auto *It = Argv.begin(); It != Argv.end(); ++It)
      *It = It->trim("\"");

    SmallString<32> Dir;
    llvm::sys::path::system_temp_directory(false, Dir);

    Entries[path(File)].push_back(
        {Dir, path(File), {Argv.begin(), Argv.end()}, "foo.o"});
  }
  void add(StringRef File, StringRef Flags = "") { add(File, "clang", Flags); }

  // Turn a unix path fragment (foo/bar.h) into a native path (C:\tmp\foo\bar.h)
  std::string path(llvm::SmallString<32> File) {
    llvm::SmallString<32> Dir;
    llvm::sys::path::system_temp_directory(false, Dir);
    llvm::sys::path::native(File);
    llvm::SmallString<64> Result;
    llvm::sys::path::append(Result, Dir, File);
    return std::string(Result.str());
  }

  MemCDB::EntryMap Entries;
};

class InterpolateTest : public MemDBTest {
protected:
  // Look up the command from a relative path, and return it in string form.
  // The input file is not included in the returned command.
  std::string getCommand(llvm::StringRef F, bool MakeNative = true) {
    auto Results =
        inferMissingCompileCommands(std::make_unique<MemCDB>(Entries))
            ->getCompileCommands(MakeNative ? path(F) : F);
    if (Results.empty())
      return "none";
    // drop the input file argument, so tests don't have to deal with path().
    EXPECT_EQ(Results[0].CommandLine.back(), MakeNative ? path(F) : F)
        << "Last arg should be the file";
    Results[0].CommandLine.pop_back();
    return llvm::join(Results[0].CommandLine, " ");
  }

  // Parse the file whose command was used out of the Heuristic string.
  std::string getProxy(llvm::StringRef F) {
    auto Results =
        inferMissingCompileCommands(std::make_unique<MemCDB>(Entries))
            ->getCompileCommands(path(F));
    if (Results.empty())
      return "none";
    StringRef Proxy = Results.front().Heuristic;
    if (!Proxy.consume_front("inferred from "))
      return "";
    // We have a proxy file, convert back to a unix relative path.
    // This is a bit messy, but we do need to test these strings somehow...
    llvm::SmallString<32> TempDir;
    llvm::sys::path::system_temp_directory(false, TempDir);
    Proxy.consume_front(TempDir);
    Proxy.consume_front(llvm::sys::path::get_separator());
    llvm::SmallString<32> Result = Proxy;
    llvm::sys::path::native(Result, llvm::sys::path::Style::posix);
    return std::string(Result.str());
  }
};

TEST_F(InterpolateTest, Nearby) {
  add("dir/foo.cpp");
  add("dir/bar.cpp");
  add("an/other/foo.cpp");

  // great: dir and name both match (prefix or full, case insensitive)
  EXPECT_EQ(getProxy("dir/f.cpp"), "dir/foo.cpp");
  EXPECT_EQ(getProxy("dir/FOO.cpp"), "dir/foo.cpp");
  // no name match. prefer matching dir, break ties by alpha
  EXPECT_EQ(getProxy("dir/a.cpp"), "dir/bar.cpp");
  // an exact name match beats one segment of directory match
  EXPECT_EQ(getProxy("some/other/bar.h"), "dir/bar.cpp");
  // two segments of directory match beat a prefix name match
  EXPECT_EQ(getProxy("an/other/b.cpp"), "an/other/foo.cpp");
  // if nothing matches at all, we still get the closest alpha match
  EXPECT_EQ(getProxy("below/some/obscure/path.cpp"), "an/other/foo.cpp");
}

TEST_F(InterpolateTest, Language) {
  add("dir/foo.cpp", "-std=c++17");
  add("dir/bar.c", "");
  add("dir/baz.cee", "-x c");
  add("dir/aux.cpp", "-std=c++17 -x objective-c++");

  // .h is ambiguous, so we add explicit language flags
  EXPECT_EQ(getCommand("foo.h"),
            "clang -D dir/foo.cpp -x c++-header -std=c++17");
  // Same thing if we have no extension. (again, we treat as header).
  EXPECT_EQ(getCommand("foo"), "clang -D dir/foo.cpp -x c++-header -std=c++17");
  // and invalid extensions.
  EXPECT_EQ(getCommand("foo.cce"),
            "clang -D dir/foo.cpp -x c++-header -std=c++17");
  // and don't add -x if the inferred language is correct.
  EXPECT_EQ(getCommand("foo.hpp"), "clang -D dir/foo.cpp -std=c++17");
  // respect -x if it's already there.
  EXPECT_EQ(getCommand("baz.h"), "clang -D dir/baz.cee -x c-header");
  // prefer a worse match with the right extension.
  EXPECT_EQ(getCommand("foo.c"), "clang -D dir/bar.c");
  Entries.erase(path(StringRef("dir/bar.c")));
  // Now we transfer across languages, so drop -std too.
  EXPECT_EQ(getCommand("foo.c"), "clang -D dir/foo.cpp");
  // Prefer -x over -std when overriding language.
  EXPECT_EQ(getCommand("aux.h"),
            "clang -D dir/aux.cpp -x objective-c++-header -std=c++17");
}

TEST_F(InterpolateTest, Strip) {
  add("dir/foo.cpp", "-o foo.o -Wall");
  // the -o option and the input file are removed, but -Wall is preserved.
  EXPECT_EQ(getCommand("dir/bar.cpp"), "clang -D dir/foo.cpp -Wall");
}

TEST_F(InterpolateTest, StripDoubleDash) {
  add("dir/foo.cpp", "-o foo.o -std=c++14 -Wall -- dir/foo.cpp");
  // input file and output option are removed
  // -Wall flag isn't
  // -std option gets re-added as the last argument before the input file
  // -- is removed as it's not necessary - the new input file doesn't start with
  // a dash
  EXPECT_EQ(getCommand("dir/bar.cpp"), "clang -D dir/foo.cpp -Wall -std=c++14");
}

TEST_F(InterpolateTest, InsertDoubleDash) {
  add("dir/foo.cpp", "-o foo.o -std=c++14 -Wall");
  EXPECT_EQ(getCommand("-dir/bar.cpp", false),
            "clang -D dir/foo.cpp -Wall -std=c++14 --");
}

TEST_F(InterpolateTest, InsertDoubleDashForClangCL) {
  add("dir/foo.cpp", "clang-cl", "/std:c++14 /W4");
  EXPECT_EQ(getCommand("/dir/bar.cpp", false),
            "clang-cl -D dir/foo.cpp /W4 /std:c++14 --");
}

TEST_F(InterpolateTest, Case) {
  add("FOO/BAR/BAZ/SHOUT.cc");
  add("foo/bar/baz/quiet.cc");
  // Case mismatches are completely ignored, so we choose the name match.
  EXPECT_EQ(getProxy("foo/bar/baz/shout.C"), "FOO/BAR/BAZ/SHOUT.cc");
}

TEST_F(InterpolateTest, Aliasing) {
  add("foo.cpp", "-faligned-new");

  // The interpolated command should keep the given flag as written, even though
  // the flag is internally represented as an alias.
  EXPECT_EQ(getCommand("foo.hpp"), "clang -D foo.cpp -faligned-new");
}

TEST_F(InterpolateTest, ClangCL) {
  add("foo.cpp", "clang-cl", "/W4");

  // Language flags should be added with CL syntax.
  EXPECT_EQ(getCommand("foo.h", false), "clang-cl -D foo.cpp /W4 /TP");
}

TEST_F(InterpolateTest, DriverModes) {
  add("foo.cpp", "clang-cl", "--driver-mode=gcc");
  add("bar.cpp", "clang", "--driver-mode=cl");

  // --driver-mode overrides should be respected.
  EXPECT_EQ(getCommand("foo.h"),
            "clang-cl -D foo.cpp --driver-mode=gcc -x c++-header");
  EXPECT_EQ(getCommand("bar.h", false),
            "clang -D bar.cpp --driver-mode=cl /TP");
}

TEST(TransferCompileCommandTest, Smoke) {
  CompileCommand Cmd;
  Cmd.Filename = "foo.cc";
  Cmd.CommandLine = {"clang", "-Wall", "foo.cc"};
  Cmd.Directory = "dir";
  CompileCommand Transferred = transferCompileCommand(std::move(Cmd), "foo.h");
  EXPECT_EQ(Transferred.Filename, "foo.h");
  EXPECT_THAT(Transferred.CommandLine,
              ElementsAre("clang", "-Wall", "-x", "c++-header", "foo.h"));
  EXPECT_EQ(Transferred.Directory, "dir");
}

TEST(CompileCommandTest, EqualityOperator) {
  CompileCommand CCRef("/foo/bar", "hello.c", {"a", "b"}, "hello.o");
  CompileCommand CCTest = CCRef;

  EXPECT_TRUE(CCRef == CCTest);
  EXPECT_FALSE(CCRef != CCTest);

  CCTest = CCRef;
  CCTest.Directory = "/foo/baz";
  EXPECT_FALSE(CCRef == CCTest);
  EXPECT_TRUE(CCRef != CCTest);

  CCTest = CCRef;
  CCTest.Filename = "bonjour.c";
  EXPECT_FALSE(CCRef == CCTest);
  EXPECT_TRUE(CCRef != CCTest);

  CCTest = CCRef;
  CCTest.CommandLine.push_back("c");
  EXPECT_FALSE(CCRef == CCTest);
  EXPECT_TRUE(CCRef != CCTest);

  CCTest = CCRef;
  CCTest.Output = "bonjour.o";
  EXPECT_FALSE(CCRef == CCTest);
  EXPECT_TRUE(CCRef != CCTest);
}

class TargetAndModeTest : public MemDBTest {
public:
  TargetAndModeTest() { llvm::InitializeAllTargetInfos(); }

protected:
  // Look up the command from a relative path, and return it in string form.
  std::string getCommand(llvm::StringRef F) {
    auto Results = inferTargetAndDriverMode(std::make_unique<MemCDB>(Entries))
                       ->getCompileCommands(path(F));
    if (Results.empty())
      return "none";
    return llvm::join(Results[0].CommandLine, " ");
  }
};

TEST_F(TargetAndModeTest, TargetAndMode) {
  add("foo.cpp", "clang-cl", "");
  add("bar.cpp", "clang++", "");

  EXPECT_EQ(getCommand("foo.cpp"),
            "clang-cl --driver-mode=cl foo.cpp -D foo.cpp");
  EXPECT_EQ(getCommand("bar.cpp"),
            "clang++ --driver-mode=g++ bar.cpp -D bar.cpp");
}

class ExpandResponseFilesTest : public MemDBTest {
public:
  ExpandResponseFilesTest() : FS(new llvm::vfs::InMemoryFileSystem) {}

protected:
  void addFile(StringRef File, StringRef Content) {
    ASSERT_TRUE(
        FS->addFile(File, 0, llvm::MemoryBuffer::getMemBufferCopy(Content)));
  }

  std::string getCommand(llvm::StringRef F) {
    auto Results = expandResponseFiles(std::make_unique<MemCDB>(Entries), FS)
                       ->getCompileCommands(path(F));
    if (Results.empty())
      return "none";
    return llvm::join(Results[0].CommandLine, " ");
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS;
};

TEST_F(ExpandResponseFilesTest, ExpandResponseFiles) {
  addFile(path(StringRef("rsp1.rsp")), "-Dflag");

  add("foo.cpp", "clang", "@rsp1.rsp");
  add("bar.cpp", "clang", "-Dflag");
  EXPECT_EQ(getCommand("foo.cpp"), "clang foo.cpp -D foo.cpp -Dflag");
  EXPECT_EQ(getCommand("bar.cpp"), "clang bar.cpp -D bar.cpp -Dflag");
}

TEST_F(ExpandResponseFilesTest, ExpandResponseFilesEmptyArgument) {
  addFile(path(StringRef("rsp1.rsp")), "-Dflag");

  add("foo.cpp", "clang", "@rsp1.rsp \"\"");
  EXPECT_EQ(getCommand("foo.cpp"), "clang foo.cpp -D foo.cpp -Dflag ");
}

} // end namespace tooling
} // end namespace clang
