//===- unittest/Tooling/CompilationDatabaseTest.cpp -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/FileMatchTrie.h"
#include "clang/Tooling/JSONCompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

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
  expected_files.push_back(PathStorage.str());
  llvm::sys::path::native("//net/dir/file2", PathStorage);
  expected_files.push_back(PathStorage.str());
  EXPECT_EQ(expected_files,
            getAllFiles("[{\"directory\":\"//net/dir\","
                        "\"command\":\"command\","
                        "\"file\":\"file1\"},"
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
                                                    StringRef JSONDatabase,
                                                    std::string &ErrorMessage) {
  std::unique_ptr<CompilationDatabase> Database(
      JSONCompilationDatabase::loadFromBuffer(JSONDatabase, ErrorMessage,
                                              JSONCommandLineSyntax::Gnu));
  if (!Database)
    return CompileCommand();
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
    return FileA.equals_lower(FileB);
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
  std::vector<std::string> CommandLine;
  CommandLine.push_back("one");
  CommandLine.push_back("two");
  FixedCompilationDatabase Database(".", CommandLine);
  StringRef FileName("source");
  std::vector<CompileCommand> Result =
    Database.getCompileCommands(FileName);
  ASSERT_EQ(1ul, Result.size());
  std::vector<std::string> ExpectedCommandLine(1, "clang-tool");
  ExpectedCommandLine.insert(ExpectedCommandLine.end(),
                             CommandLine.begin(), CommandLine.end());
  ExpectedCommandLine.push_back("source");
  EXPECT_EQ(".", Result[0].Directory);
  EXPECT_EQ(FileName, Result[0].Filename);
  EXPECT_EQ(ExpectedCommandLine, Result[0].CommandLine);
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

TEST(ParseFixedCompilationDatabase, ReturnsNullOnEmptyArgumentList) {
  int Argc = 0;
  std::unique_ptr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, nullptr));
  EXPECT_FALSE(Database);
  EXPECT_EQ(0, Argc);
}

TEST(ParseFixedCompilationDatabase, ReturnsNullWithoutDoubleDash) {
  int Argc = 2;
  const char *Argv[] = { "1", "2" };
  std::unique_ptr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv));
  EXPECT_FALSE(Database);
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, ReturnsArgumentsAfterDoubleDash) {
  int Argc = 5;
  const char *Argv[] = {
    "1", "2", "--\0no-constant-folding", "-DDEF3", "-DDEF4"
  };
  std::unique_ptr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv));
  ASSERT_TRUE((bool)Database);
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  std::vector<std::string> CommandLine;
  CommandLine.push_back("clang-tool");
  CommandLine.push_back("-DDEF3");
  CommandLine.push_back("-DDEF4");
  CommandLine.push_back("source");
  ASSERT_EQ(CommandLine, Result[0].CommandLine);
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, ReturnsEmptyCommandLine) {
  int Argc = 3;
  const char *Argv[] = { "1", "2", "--\0no-constant-folding" };
  std::unique_ptr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv));
  ASSERT_TRUE((bool)Database);
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  std::vector<std::string> CommandLine;
  CommandLine.push_back("clang-tool");
  CommandLine.push_back("source");
  ASSERT_EQ(CommandLine, Result[0].CommandLine);
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, HandlesPositionalArgs) {
  const char *Argv[] = {"1", "2", "--", "-c", "somefile.cpp", "-DDEF3"};
  int Argc = sizeof(Argv) / sizeof(char*);
  std::unique_ptr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv));
  ASSERT_TRUE((bool)Database);
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  std::vector<std::string> Expected;
  Expected.push_back("clang-tool");
  Expected.push_back("-c");
  Expected.push_back("-DDEF3");
  Expected.push_back("source");
  ASSERT_EQ(Expected, Result[0].CommandLine);
  EXPECT_EQ(2, Argc);
}

TEST(ParseFixedCompilationDatabase, HandlesArgv0) {
  const char *Argv[] = {"1", "2", "--", "mytool", "somefile.cpp"};
  int Argc = sizeof(Argv) / sizeof(char*);
  std::unique_ptr<FixedCompilationDatabase> Database(
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv));
  ASSERT_TRUE((bool)Database);
  std::vector<CompileCommand> Result =
    Database->getCompileCommands("source");
  ASSERT_EQ(1ul, Result.size());
  ASSERT_EQ(".", Result[0].Directory);
  std::vector<std::string> Expected;
  Expected.push_back("clang-tool");
  Expected.push_back("source");
  ASSERT_EQ(Expected, Result[0].CommandLine);
  EXPECT_EQ(2, Argc);
}

} // end namespace tooling
} // end namespace clang
