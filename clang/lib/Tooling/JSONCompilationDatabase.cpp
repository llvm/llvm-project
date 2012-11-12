//===--- JSONCompilationDatabase.cpp - ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains the implementation of the JSONCompilationDatabase.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/JSONCompilationDatabase.h"

#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/CompilationDatabasePluginRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/system_error.h"

namespace clang {
namespace tooling {

namespace {

/// \brief A parser for escaped strings of command line arguments.
///
/// Assumes \-escaping for quoted arguments (see the documentation of
/// unescapeCommandLine(...)).
class CommandLineArgumentParser {
 public:
  CommandLineArgumentParser(StringRef CommandLine)
      : Input(CommandLine), Position(Input.begin()-1) {}

  std::vector<std::string> parse() {
    bool HasMoreInput = true;
    while (HasMoreInput && nextNonWhitespace()) {
      std::string Argument;
      HasMoreInput = parseStringInto(Argument);
      CommandLine.push_back(Argument);
    }
    return CommandLine;
  }

 private:
  // All private methods return true if there is more input available.

  bool parseStringInto(std::string &String) {
    do {
      if (*Position == '"') {
        if (!parseQuotedStringInto(String)) return false;
      } else {
        if (!parseFreeStringInto(String)) return false;
      }
    } while (*Position != ' ');
    return true;
  }

  bool parseQuotedStringInto(std::string &String) {
    if (!next()) return false;
    while (*Position != '"') {
      if (!skipEscapeCharacter()) return false;
      String.push_back(*Position);
      if (!next()) return false;
    }
    return next();
  }

  bool parseFreeStringInto(std::string &String) {
    do {
      if (!skipEscapeCharacter()) return false;
      String.push_back(*Position);
      if (!next()) return false;
    } while (*Position != ' ' && *Position != '"');
    return true;
  }

  bool skipEscapeCharacter() {
    if (*Position == '\\') {
      return next();
    }
    return true;
  }

  bool nextNonWhitespace() {
    do {
      if (!next()) return false;
    } while (*Position == ' ');
    return true;
  }

  bool next() {
    ++Position;
    return Position != Input.end();
  }

  const StringRef Input;
  StringRef::iterator Position;
  std::vector<std::string> CommandLine;
};

std::vector<std::string> unescapeCommandLine(
    StringRef EscapedCommandLine) {
  CommandLineArgumentParser parser(EscapedCommandLine);
  return parser.parse();
}

} // end namespace

class JSONCompilationDatabasePlugin : public CompilationDatabasePlugin {
  virtual CompilationDatabase *loadFromDirectory(
      StringRef Directory, std::string &ErrorMessage) {
    llvm::SmallString<1024> JSONDatabasePath(Directory);
    llvm::sys::path::append(JSONDatabasePath, "compile_commands.json");
    llvm::OwningPtr<CompilationDatabase> Database(
        JSONCompilationDatabase::loadFromFile(JSONDatabasePath, ErrorMessage));
    if (!Database)
      return NULL;
    return Database.take();
  }
};

// Register the JSONCompilationDatabasePlugin with the
// CompilationDatabasePluginRegistry using this statically initialized variable.
static CompilationDatabasePluginRegistry::Add<JSONCompilationDatabasePlugin>
X("json-compilation-database", "Reads JSON formatted compilation databases");

// This anchor is used to force the linker to link in the generated object file
// and thus register the JSONCompilationDatabasePlugin.
volatile int JSONAnchorSource = 0;

JSONCompilationDatabase *
JSONCompilationDatabase::loadFromFile(StringRef FilePath,
                                      std::string &ErrorMessage) {
  llvm::OwningPtr<llvm::MemoryBuffer> DatabaseBuffer;
  llvm::error_code Result =
    llvm::MemoryBuffer::getFile(FilePath, DatabaseBuffer);
  if (Result != 0) {
    ErrorMessage = "Error while opening JSON database: " + Result.message();
    return NULL;
  }
  llvm::OwningPtr<JSONCompilationDatabase> Database(
    new JSONCompilationDatabase(DatabaseBuffer.take()));
  if (!Database->parse(ErrorMessage))
    return NULL;
  return Database.take();
}

JSONCompilationDatabase *
JSONCompilationDatabase::loadFromBuffer(StringRef DatabaseString,
                                        std::string &ErrorMessage) {
  llvm::OwningPtr<llvm::MemoryBuffer> DatabaseBuffer(
      llvm::MemoryBuffer::getMemBuffer(DatabaseString));
  llvm::OwningPtr<JSONCompilationDatabase> Database(
    new JSONCompilationDatabase(DatabaseBuffer.take()));
  if (!Database->parse(ErrorMessage))
    return NULL;
  return Database.take();
}

std::vector<CompileCommand>
JSONCompilationDatabase::getCompileCommands(StringRef FilePath) const {
  llvm::SmallString<128> NativeFilePath;
  llvm::sys::path::native(FilePath, NativeFilePath);
  std::vector<StringRef> PossibleMatches;
  std::string Error;
  llvm::raw_string_ostream ES(Error);
  StringRef Match = MatchTrie.findEquivalent(NativeFilePath.str(), ES);
  if (Match.empty()) {
    if (Error.empty())
      Error = "No match found.";
    llvm::outs() << Error << "\n";
    return std::vector<CompileCommand>();
  }
  llvm::StringMap< std::vector<CompileCommandRef> >::const_iterator
    CommandsRefI = IndexByFile.find(Match);
  if (CommandsRefI == IndexByFile.end())
    return std::vector<CompileCommand>();
  const std::vector<CompileCommandRef> &CommandsRef = CommandsRefI->getValue();
  std::vector<CompileCommand> Commands;
  for (int I = 0, E = CommandsRef.size(); I != E; ++I) {
    llvm::SmallString<8> DirectoryStorage;
    llvm::SmallString<1024> CommandStorage;
    Commands.push_back(CompileCommand(
      // FIXME: Escape correctly:
      CommandsRef[I].first->getValue(DirectoryStorage),
      unescapeCommandLine(CommandsRef[I].second->getValue(CommandStorage))));
  }
  return Commands;
}

std::vector<std::string>
JSONCompilationDatabase::getAllFiles() const {
  std::vector<std::string> Result;

  llvm::StringMap< std::vector<CompileCommandRef> >::const_iterator
    CommandsRefI = IndexByFile.begin();
  const llvm::StringMap< std::vector<CompileCommandRef> >::const_iterator
    CommandsRefEnd = IndexByFile.end();
  for (; CommandsRefI != CommandsRefEnd; ++CommandsRefI) {
    Result.push_back(CommandsRefI->first().str());
  }

  return Result;
}

bool JSONCompilationDatabase::parse(std::string &ErrorMessage) {
  llvm::yaml::document_iterator I = YAMLStream.begin();
  if (I == YAMLStream.end()) {
    ErrorMessage = "Error while parsing YAML.";
    return false;
  }
  llvm::yaml::Node *Root = I->getRoot();
  if (Root == NULL) {
    ErrorMessage = "Error while parsing YAML.";
    return false;
  }
  llvm::yaml::SequenceNode *Array =
    llvm::dyn_cast<llvm::yaml::SequenceNode>(Root);
  if (Array == NULL) {
    ErrorMessage = "Expected array.";
    return false;
  }
  for (llvm::yaml::SequenceNode::iterator AI = Array->begin(),
                                          AE = Array->end();
       AI != AE; ++AI) {
    llvm::yaml::MappingNode *Object =
      llvm::dyn_cast<llvm::yaml::MappingNode>(&*AI);
    if (Object == NULL) {
      ErrorMessage = "Expected object.";
      return false;
    }
    llvm::yaml::ScalarNode *Directory = NULL;
    llvm::yaml::ScalarNode *Command = NULL;
    llvm::yaml::ScalarNode *File = NULL;
    for (llvm::yaml::MappingNode::iterator KVI = Object->begin(),
                                           KVE = Object->end();
         KVI != KVE; ++KVI) {
      llvm::yaml::Node *Value = (*KVI).getValue();
      if (Value == NULL) {
        ErrorMessage = "Expected value.";
        return false;
      }
      llvm::yaml::ScalarNode *ValueString =
        llvm::dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (ValueString == NULL) {
        ErrorMessage = "Expected string as value.";
        return false;
      }
      llvm::yaml::ScalarNode *KeyString =
        llvm::dyn_cast<llvm::yaml::ScalarNode>((*KVI).getKey());
      if (KeyString == NULL) {
        ErrorMessage = "Expected strings as key.";
        return false;
      }
      llvm::SmallString<8> KeyStorage;
      if (KeyString->getValue(KeyStorage) == "directory") {
        Directory = ValueString;
      } else if (KeyString->getValue(KeyStorage) == "command") {
        Command = ValueString;
      } else if (KeyString->getValue(KeyStorage) == "file") {
        File = ValueString;
      } else {
        ErrorMessage = ("Unknown key: \"" +
                        KeyString->getRawValue() + "\"").str();
        return false;
      }
    }
    if (!File) {
      ErrorMessage = "Missing key: \"file\".";
      return false;
    }
    if (!Command) {
      ErrorMessage = "Missing key: \"command\".";
      return false;
    }
    if (!Directory) {
      ErrorMessage = "Missing key: \"directory\".";
      return false;
    }
    llvm::SmallString<8> FileStorage;
    StringRef FileName = File->getValue(FileStorage);
    llvm::SmallString<128> NativeFilePath;
    if (llvm::sys::path::is_relative(FileName)) {
      llvm::SmallString<8> DirectoryStorage;
      llvm::SmallString<128> AbsolutePath(
          Directory->getValue(DirectoryStorage));
      llvm::sys::path::append(AbsolutePath, FileName);
      llvm::sys::path::native(AbsolutePath.str(), NativeFilePath);
    } else {
      llvm::sys::path::native(FileName, NativeFilePath);
    }
    IndexByFile[NativeFilePath].push_back(
        CompileCommandRef(Directory, Command));
    MatchTrie.insert(NativeFilePath.str());
  }
  return true;
}

} // end namespace tooling
} // end namespace clang
