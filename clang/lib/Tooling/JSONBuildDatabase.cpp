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
#include <string>
#include <system_error>
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
  auto *Array = dyn_cast<llvm::yaml::SequenceNode>(Root);
  if (!Array) {
    ErrorMessage = "Expected array.";
    return false;
  }
  for (auto &NextObject : *Array) {
    auto *Object = dyn_cast<llvm::yaml::MappingNode>(&NextObject);
    if (!Object) {
      ErrorMessage = "Expected object.";
      return false;
    }
    llvm::yaml::ScalarNode *Directory = nullptr;
    std::optional<std::vector<llvm::yaml::ScalarNode *>> Command;
    llvm::yaml::ScalarNode *File = nullptr;
    llvm::yaml::ScalarNode *Output = nullptr;
    for (auto &NextKeyValue : *Object) {
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
      auto *ValueString = dyn_cast<llvm::yaml::ScalarNode>(Value);
      auto *SequenceString = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (KeyValue == "arguments") {
        if (!SequenceString) {
          ErrorMessage = "Expected sequence as value.";
          return false;
        }
        Command = std::vector<llvm::yaml::ScalarNode *>();
        for (auto &Argument : *SequenceString) {
          auto *Scalar = dyn_cast<llvm::yaml::ScalarNode>(&Argument);
          if (!Scalar) {
            ErrorMessage = "Only strings are allowed in 'arguments'.";
            return false;
          }
          Command->push_back(Scalar);
        }
      } else {
        if (!ValueString) {
          ErrorMessage = "Expected string as value.";
          return false;
        }
        if (KeyValue == "directory") {
          Directory = ValueString;
        } else if (KeyValue == "command") {
          if (!Command)
            Command = std::vector<llvm::yaml::ScalarNode *>(1, ValueString);
        } else if (KeyValue == "file") {
          File = ValueString;
        } else if (KeyValue == "output") {
          Output = ValueString;
        } else {
          ErrorMessage =
              ("Unknown key: \"" + KeyString->getRawValue() + "\"").str();
          return false;
        }
      }
    }
    if (!File) {
      ErrorMessage = "Missing key: \"file\".";
      return false;
    }
    if (!Command) {
      ErrorMessage = "Missing key: \"command\" or \"arguments\".";
      return false;
    }
    if (!Directory) {
      ErrorMessage = "Missing key: \"directory\".";
      return false;
    }
    SmallString<8> FileStorage;
    StringRef FileName = File->getValue(FileStorage);
    SmallString<128> NativeFilePath;
    if (llvm::sys::path::is_relative(FileName)) {
      SmallString<8> DirectoryStorage;
      SmallString<128> AbsolutePath(Directory->getValue(DirectoryStorage));
      llvm::sys::path::append(AbsolutePath, FileName);
      llvm::sys::path::native(AbsolutePath, NativeFilePath);
    } else {
      llvm::sys::path::native(FileName, NativeFilePath);
    }
    llvm::sys::path::remove_dots(NativeFilePath, /*remove_dot_dot=*/true);
    auto Cmd = CompileCommandRef(Directory, File, *Command, Output);
    IndexByFile[NativeFilePath].push_back(Cmd);
    AllCommands.push_back(Cmd);
    MatchTrie.insert(NativeFilePath);
  }
  return true;
}
