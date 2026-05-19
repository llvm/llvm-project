//===- JSONBuildDatabase.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains the implementation of the JSONBuildDatabase.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/JSONBuildDatabase.h"
#include "clang/Basic/LLVM.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/CompilationDatabasePluginRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <cassert>
#include <memory>
#include <optional>
#include <signal.h>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace clang;
using namespace tooling;

namespace {
// This plugin locates a nearby compile_command.json file, and also infers
// compile commands for files not present in the database.
class JSONBuildDatabasePlugin : public CompilationDatabasePlugin {
  std::unique_ptr<CompilationDatabase>
  loadFromDirectory(StringRef Directory, std::string &ErrorMessage) override {
    SmallString<1024> JSONDatabasePath(Directory);
    llvm::sys::path::append(JSONDatabasePath, "compile_commands.json");
    auto Base = JSONBuildDatabase::loadFromFile(JSONDatabasePath, ErrorMessage);
    return Base ? inferTargetAndDriverMode(
                      inferMissingCompileCommands(expandResponseFiles(
                          std::move(Base), llvm::vfs::getRealFileSystem())))
                : nullptr;
  }
};

} // namespace

// Register the JSONBuildDatabasePlugin with the
// CompilationDatabasePluginRegistry using this statically initialized variable.
static CompilationDatabasePluginRegistry::Add<JSONBuildDatabasePlugin>
    X("json-build-database", "Reads JSON formatted build databases");

namespace clang {
namespace tooling {

// This anchor is used to force the linker to link in the generated object file
// and thus register the JSONBuildDatabasePlugin.
volatile int JSONBuildAnchorSource = 0;

} // namespace tooling
} // namespace clang

std::unique_ptr<JSONBuildDatabase>
JSONBuildDatabase::loadFromFile(StringRef FilePath, std::string &ErrorMessage) {
  // Don't mmap: if we're a long-lived process, the build system may overwrite.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> DatabaseBuffer =
      llvm::MemoryBuffer::getFile(FilePath, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/true,
                                  /*IsVolatile=*/true);
  if (std::error_code Result = DatabaseBuffer.getError()) {
    ErrorMessage = "Error while opening JSON database: " + Result.message();
    return nullptr;
  }
  std::unique_ptr<JSONBuildDatabase> Database(
      new JSONBuildDatabase(std::move(*DatabaseBuffer)));
  if (!Database->parse(ErrorMessage))
    return nullptr;
  return Database;
}

std::unique_ptr<JSONBuildDatabase>
JSONBuildDatabase::loadFromBuffer(StringRef DatabaseString,
                                  std::string &ErrorMessage) {
  std::unique_ptr<llvm::MemoryBuffer> DatabaseBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(DatabaseString));
  std::unique_ptr<JSONBuildDatabase> Database(
      new JSONBuildDatabase(std::move(DatabaseBuffer)));
  if (!Database->parse(ErrorMessage))
    return nullptr;
  return Database;
}

std::vector<CompileCommand>
JSONBuildDatabase::getCompileCommands(StringRef FilePath) const {
  SmallString<128> NativeFilePath;
  llvm::sys::path::native(FilePath, NativeFilePath);

  std::string Error;
  llvm::raw_string_ostream ES(Error);
  StringRef Match = MatchTrie.findEquivalent(NativeFilePath, ES);
  if (Match.empty())
    return {};
  const auto CommandsRefI = IndexByFile.find(Match);
  if (CommandsRefI == IndexByFile.end())
    return {};
  std::vector<CompileCommand> Commands;
  getCommands(CommandsRefI->getValue(), Commands);
  return Commands;
}

std::vector<std::string> JSONBuildDatabase::getAllFiles() const {
  std::vector<std::string> Result;
  for (const auto &CommandRef : IndexByFile)
    Result.push_back(CommandRef.first().str());
  return Result;
}

std::vector<CompileCommand> JSONBuildDatabase::getAllCompileCommands() const {
  std::vector<CompileCommand> Commands;
  getCommands(AllCommands, Commands);
  return Commands;
}

static llvm::StringRef stripExecutableExtension(llvm::StringRef Name) {
  Name.consume_back(".exe");
  return Name;
}

// There are compiler-wrappers (ccache, distcc) that take the "real"
// compiler as an argument, e.g. distcc gcc -O3 foo.c.
// These end up in compile_commands.json when people set CC="distcc gcc".
// Clang's driver doesn't understand this, so we need to unwrap.
static bool unwrapCommand(std::vector<std::string> &Args) {
  if (Args.size() < 2)
    return false;
  StringRef Wrapper =
      stripExecutableExtension(llvm::sys::path::filename(Args.front()));
  if (Wrapper == "distcc" || Wrapper == "ccache" || Wrapper == "sccache") {
    // Most of these wrappers support being invoked 3 ways:
    // `distcc g++ file.c` This is the mode we're trying to match.
    //                     We need to drop `distcc`.
    // `distcc file.c`     This acts like compiler is cc or similar.
    //                     Clang's driver can handle this, no change needed.
    // `g++ file.c`        g++ is a symlink to distcc.
    //                     We don't even notice this case, and all is well.
    //
    // We need to distinguish between the first and second case.
    // The wrappers themselves don't take flags, so Args[1] is a compiler flag,
    // an input file, or a compiler. Inputs have extensions, compilers don't.
    bool HasCompiler =
        (Args[1][0] != '-') &&
        !llvm::sys::path::has_extension(stripExecutableExtension(Args[1]));
    if (HasCompiler) {
      Args.erase(Args.begin());
      return true;
    }
    // If !HasCompiler, wrappers act like GCC. Fine: so do we.
  }
  return false;
}

static std::vector<std::string>
nodeToCommandLine(const std::vector<llvm::yaml::ScalarNode *> &Nodes) {
  SmallString<1024> Storage;
  std::vector<std::string> Arguments;
  for (const auto *Node : Nodes)
    Arguments.push_back(std::string(Node->getValue(Storage)));
  // There may be multiple wrappers: using distcc and ccache together is common.
  while (unwrapCommand(Arguments))
    ;
  return Arguments;
}

void JSONBuildDatabase::getCommands(
    ArrayRef<CompileCommandRef> CommandsRef,
    std::vector<CompileCommand> &Commands) const {
  for (const auto &CommandRef : CommandsRef) {
    SmallString<8> DirectoryStorage;
    SmallString<32> FilenameStorage;
    SmallString<32> OutputStorage;
    auto Output = std::get<3>(CommandRef);
    Commands.emplace_back(std::get<0>(CommandRef)->getValue(DirectoryStorage),
                          std::get<1>(CommandRef)->getValue(FilenameStorage),
                          nodeToCommandLine(std::get<2>(CommandRef)),
                          Output ? Output->getValue(OutputStorage) : "");
  }
}

bool JSONBuildDatabase::parse(std::string &ErrorMessage) {
  llvm::yaml::document_iterator I = YAMLStream.begin();
  if (I == YAMLStream.end()) {
    ErrorMessage = "Error while parsing YAML.";
    return false;
  }
  llvm::yaml::Node *Root = I->getRoot();
  if (!Root) {
    ErrorMessage = "Error while parsing YAML.";
    return false;
  }
  auto *RootObject = dyn_cast<llvm::yaml::MappingNode>(Root);
  if (!RootObject) {
    ErrorMessage = "Expected object at root.";
    return false;
  }
  return parseRoot(ErrorMessage, RootObject);
}

bool JSONBuildDatabase::parseRoot(std::string &ErrorMessage,
                                  llvm::yaml::MappingNode *RootObject) {
  llvm::yaml::ScalarNode *Version = nullptr;
  llvm::yaml::ScalarNode *Revision = nullptr;
  llvm::yaml::SequenceNode *Sets = nullptr;
  for (auto &NextKeyValue : *RootObject) {
    auto *KeyString =
        dyn_cast_if_present<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString) {
      ErrorMessage = "Expected strings as key.";
      return false;
    }
    SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    llvm::yaml::Node *Value = NextKeyValue.getValue();
    if (!Value) {
      ErrorMessage = "Expected value.";
      return false;
    }
    if (KeyValue == "version") {
      Version = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Version) {
        ErrorMessage = "Expected string as value for \"version\".";
        return false;
      }
    } else if (KeyValue == "revision") {
      Revision = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Revision) {
        ErrorMessage = "Expected string as value for \"revision\".";
        return false;
      }
    } else if (KeyValue == "sets") {
      Sets = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!Sets) {
        ErrorMessage = "Expected array as value for \"sets\".";
        return false;
      }
      for (auto &NextObject : *Sets) {
        auto *SetObject = dyn_cast<llvm::yaml::MappingNode>(&NextObject);
        if (!RootObject) {
          ErrorMessage = "Expected sets item as object.";
          return false;
        }
        if (!parseSet(ErrorMessage, SetObject)) {
          return false;
        }
      }
    } else {
      ErrorMessage =
          ("Unknown key in root: \"" + KeyString->getRawValue() + "\"").str();
      return false;
    }
  }
  // Check required fields
  if (!Version) {
    ErrorMessage = "Missing key in root: \"version\".";
    return false;
  }
  if (!Sets) {
    ErrorMessage = "Missing key in root: \"sets\".";
    return false;
  }
  // Check compatible version
  if (Version->getRawValue() != "1") {
    ErrorMessage =
        ("Unsupported version: \"" + Version->getRawValue() + "\"").str();
    return false;
  }
  return true;
}

bool JSONBuildDatabase::parseSet(std::string &ErrorMessage,
                                 llvm::yaml::MappingNode *SetObject) {
  llvm::yaml::SequenceNode *BaselineArguments = nullptr;
  llvm::yaml::ScalarNode *FamilyName = nullptr;
  llvm::yaml::ScalarNode *Name = nullptr;
  llvm::yaml::SequenceNode *VisibleSets = nullptr;
  llvm::yaml::SequenceNode *TUs = nullptr;
  for (auto &NextKeyValue : *SetObject) {
    auto *KeyString =
        dyn_cast_if_present<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString) {
      ErrorMessage = "Expected strings as key.";
      return false;
    }
    SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    llvm::yaml::Node *Value = NextKeyValue.getValue();
    if (!Value) {
      ErrorMessage = "Expected value.";
      return false;
    }
    if (KeyValue == "baseline-arguments") {
      BaselineArguments = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!BaselineArguments) {
        ErrorMessage = "Expected array as value for \"version\".";
        return false;
      }
    } else if (KeyValue == "family-name") {
      FamilyName = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!FamilyName) {
        ErrorMessage = "Expected string as value for \"family-name\".";
        return false;
      }
    } else if (KeyValue == "name") {
      Name = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Name) {
        ErrorMessage = "Expected string as value for \"name\".";
        return false;
      }
    } else if (KeyValue == "visible-sets") {
      VisibleSets = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!VisibleSets) {
        ErrorMessage = "Expected array as value for \"visible-sets\".";
        return false;
      }
    } else if (KeyValue == "translation-units") {
      TUs = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!TUs) {
        ErrorMessage = "Expected array as value for \"translation-units\".";
        return false;
      }
      for (auto &NextObject : *TUs) {
        auto *TUObject = dyn_cast<llvm::yaml::MappingNode>(&NextObject);
        if (!TUObject) {
          ErrorMessage = "Expected translation-units item as object.";
          return false;
        }
        if (!parseTU(ErrorMessage, TUObject)) {
          return false;
        }
      }
    } else {
      ErrorMessage =
          ("Unknown key in set: \"" + KeyString->getRawValue() + "\"").str();
      return false;
    }
  }
  // Check required fields
  if (!BaselineArguments) {
    ErrorMessage = "Missing key in set: \"baseline-arguments\".";
    return false;
  }
  if (!FamilyName) {
    ErrorMessage = "Missing key in set: \"family-name\".";
    return false;
  }
  if (!Name) {
    ErrorMessage = "Missing key in set: \"name\".";
    return false;
  }
  if (!TUs) {
    ErrorMessage = "Missing key in set: \"translation-units\".";
    return false;
  }
  return true;
}

bool JSONBuildDatabase::parseTU(std::string &ErrorMessage,
                                llvm::yaml::MappingNode *TUObject) {
  llvm::yaml::SequenceNode *Arguments = nullptr;
  std::vector<llvm::yaml::ScalarNode *> Command;
  llvm::yaml::ScalarNode *Language = nullptr;
  llvm::yaml::SequenceNode *LocalArguments = nullptr;
  llvm::yaml::ScalarNode *WorkDirectory = nullptr;
  llvm::yaml::ScalarNode *Private = nullptr;
  llvm::yaml::ScalarNode *Source = nullptr;
  llvm::yaml::ScalarNode *Object = nullptr;
  llvm::yaml::MappingNode *Provides = nullptr;
  llvm::yaml::SequenceNode *Requires = nullptr;
  for (auto &NextKeyValue : *TUObject) {
    auto *KeyString =
        dyn_cast_if_present<llvm::yaml::ScalarNode>(NextKeyValue.getKey());
    if (!KeyString) {
      ErrorMessage = "Expected strings as key.";
      return false;
    }
    SmallString<10> KeyStorage;
    StringRef KeyValue = KeyString->getValue(KeyStorage);
    llvm::yaml::Node *Value = NextKeyValue.getValue();
    if (!Value) {
      ErrorMessage = "Expected value.";
      return false;
    }
    if (KeyValue == "arguments") {
      Arguments = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!Arguments) {
        ErrorMessage = "Expected array as value for \"arguments\".";
        return false;
      }
      for (auto &Argument : *Arguments) {
        auto *Scalar = dyn_cast<llvm::yaml::ScalarNode>(&Argument);
        if (!Scalar) {
          ErrorMessage = "Only strings are allowed in 'arguments'.";
          return false;
        }
        Command.push_back(Scalar);
      }
    } else if (KeyValue == "language") {
      Language = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Language) {
        ErrorMessage = "Expected string as value for \"language\".";
        return false;
      }
    } else if (KeyValue == "local-arguments") {
      LocalArguments = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!LocalArguments) {
        ErrorMessage = "Expected array as value for \"local-arguments\".";
        return false;
      }
    } else if (KeyValue == "work-directory") {
      WorkDirectory = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!WorkDirectory) {
        ErrorMessage = "Expected string as value for \"work-directory\".";
        return false;
      }
    } else if (KeyValue == "private") {
      Private = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Private) {
        ErrorMessage = "Expected string as value for \"private\".";
        return false;
      }
    } else if (KeyValue == "source") {
      Source = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Source) {
        ErrorMessage = "Expected string as value for \"source\".";
        return false;
      }
    } else if (KeyValue == "object") {
      Object = dyn_cast<llvm::yaml::ScalarNode>(Value);
      if (!Object) {
        ErrorMessage = "Expected string as value for \"object\".";
        return false;
      }
    } else if (KeyValue == "provides") {
      Provides = dyn_cast<llvm::yaml::MappingNode>(Value);
      if (!Provides) {
        ErrorMessage = "Expected object as value for \"provides\".";
        return false;
      }
    } else if (KeyValue == "requires") {
      Requires = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!Requires) {
        ErrorMessage = "Expected array as value for \"requires\".";
        return false;
      }
    } else {
      ErrorMessage = ("Unknown key in translation-unit: \"" +
                      KeyString->getRawValue() + "\"")
                         .str();
      return false;
    }
  }
  // Check required fields
  if (!Source) {
    ErrorMessage = "Missing key in translation-unit: \"source\".";
    return false;
  }
  if (!Language) {
    ErrorMessage = "Missing key in translation-unit: \"language\".";
    return false;
  }
  if (!Arguments) {
    ErrorMessage = "Missing key in translation-unit: \"arguments\".";
    return false;
  }
  SmallString<8> FileStorage;
  StringRef FileName = Source->getValue(FileStorage);
  SmallString<128> NativeFilePath;
  if (llvm::sys::path::is_relative(FileName)) {
    SmallString<8> DirectoryStorage;
    SmallString<128> AbsolutePath(WorkDirectory->getValue(DirectoryStorage));
    llvm::sys::path::append(AbsolutePath, FileName);
    llvm::sys::path::native(AbsolutePath, NativeFilePath);
  } else {
    llvm::sys::path::native(FileName, NativeFilePath);
  }
  llvm::sys::path::remove_dots(NativeFilePath, /*remove_dot_dot=*/true);
  auto Cmd = CompileCommandRef(WorkDirectory, Source, Command, Object);

  IndexByFile[NativeFilePath].push_back(Cmd);
  AllCommands.push_back(Cmd);
  MatchTrie.insert(NativeFilePath);

  return true;
}
