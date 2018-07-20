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
  std::string ErrorMsg;
  std::unique_ptr<FixedCompilationDatabase> Database =
      FixedCompilationDatabase::loadFromCommandLine(Argc, Argv, ErrorMsg);
  ASSERT_TRUE((bool)Database);
  ASSERT_TRUE(ErrorMsg.empty());
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
  Expected.push_back("clang-tool");
  Expected.push_back("-c");
  Expected.push_back("-DDEF3");
  Expected.push_back("source");
  ASSERT_EQ(Expected, Result[0].CommandLine);
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
  std::vector<std::string> Expected;
  Expected.push_back("clang-tool");
  Expected.push_back("-fsyntax-only");
  Expected.push_back("-DDEF3");
  Expected.push_back("source");
  ASSERT_EQ(Expected, Result[0].CommandLine);
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
  Expected.push_back("clang-tool");
  Expected.push_back("source");
  ASSERT_EQ(Expected, Result[0].CommandLine);
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
      Result.push_back(Entry.first());
    return Result;
  }
};

class InterpolateTest : public ::testing::Test {
protected:
  // Adds an entry to the underlying compilation database.
  // A flag is injected: -D <File>, so the command used can be identified.
  void add(llvm::StringRef File, llvm::StringRef Flags = "") {
    llvm::SmallVector<StringRef, 8> Argv = {"clang", File, "-D", File};
    llvm::SplitString(Flags, Argv);
    llvm::SmallString<32> Dir;
    llvm::sys::path::system_temp_directory(false, Dir);
    Entries[path(File)].push_back(
        {Dir, path(File), {Argv.begin(), Argv.end()}, "foo.o"});
  }

  // Turn a unix path fragment (foo/bar.h) into a native path (C:\tmp\foo\bar.h)
  std::string path(llvm::SmallString<32> File) {
    llvm::SmallString<32> Dir;
    llvm::sys::path::system_temp_directory(false, Dir);
    llvm::sys::path::native(File);
    llvm::SmallString<64> Result;
    llvm::sys::path::append(Result, Dir, File);
    return Result.str();
  }

  // Look up the command from a relative path, and return it in string form.
  // The input file is not included in the returned command.
  std::string getCommand(llvm::StringRef F) {
    auto Results =
        inferMissingCompileCommands(llvm::make_unique<MemCDB>(Entries))
            ->getCompileCommands(path(F));
    if (Results.empty())
      return "none";
    // drop the input file argument, so tests don't have to deal with path().
    EXPECT_EQ(Results[0].CommandLine.back(), path(F))
        << "Last arg should be the file";
    Results[0].CommandLine.pop_back();
    return llvm::join(Results[0].CommandLine, " ");
  }

  MemCDB::EntryMap Entries;
};

TEST_F(InterpolateTest, Nearby) {
  add("dir/foo.cpp");
  add("dir/bar.cpp");
  add("an/other/foo.cpp");

  // great: dir and name both match (prefix or full, case insensitive)
  EXPECT_EQ(getCommand("dir/f.cpp"), "clang -D dir/foo.cpp");
  EXPECT_EQ(getCommand("dir/FOO.cpp"), "clang -D dir/foo.cpp");
  // no name match. prefer matching dir, break ties by alpha
  EXPECT_EQ(getCommand("dir/a.cpp"), "clang -D dir/bar.cpp");
  // an exact name match beats one segment of directory match
  EXPECT_EQ(getCommand("some/other/bar.h"),
            "clang -D dir/bar.cpp -x c++-header");
  // two segments of directory match beat a prefix name match
  EXPECT_EQ(getCommand("an/other/b.cpp"), "clang -D an/other/foo.cpp");
  // if nothing matches at all, we still get the closest alpha match
  EXPECT_EQ(getCommand("below/some/obscure/path.cpp"),
            "clang -D an/other/foo.cpp");
}

TEST_F(InterpolateTest, Language) {
  add("dir/foo.cpp", "-std=c++17");
  add("dir/baz.cee", "-x c");

  // .h is ambiguous, so we add explicit language flags
  EXPECT_EQ(getCommand("foo.h"),
            "clang -D dir/foo.cpp -x c++-header -std=c++17");
  // and don't add -x if the inferred language is correct.
  EXPECT_EQ(getCommand("foo.hpp"), "clang -D dir/foo.cpp -std=c++17");
  // respect -x if it's already there.
  EXPECT_EQ(getCommand("baz.h"), "clang -D dir/baz.cee -x c-header");
  // prefer a worse match with the right language
  EXPECT_EQ(getCommand("foo.c"), "clang -D dir/baz.cee");
  Entries.erase(path(StringRef("dir/baz.cee")));
  // Now we transfer across languages, so drop -std too.
  EXPECT_EQ(getCommand("foo.c"), "clang -D dir/foo.cpp");
}

TEST_F(InterpolateTest, Strip) {
  add("dir/foo.cpp", "-o foo.o -Wall");
  // the -o option and the input file are removed, but -Wall is preserved.
  EXPECT_EQ(getCommand("dir/bar.cpp"), "clang -D dir/foo.cpp -Wall");
}

TEST_F(InterpolateTest, Case) {
  add("FOO/BAR/BAZ/SHOUT.cc");
  add("foo/bar/baz/quiet.cc");
  // Case mismatches are completely ignored, so we choose the name match.
  EXPECT_EQ(getCommand("foo/bar/baz/shout.C"), "clang -D FOO/BAR/BAZ/SHOUT.cc");
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

} // end namespace tooling
} // end namespace clang
