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
#include <unistd.h>
#include <utility>
#include <vector>

using namespace clang;
using namespace tooling;

namespace {
// This plugin locates a nearby build_database.json file, and also infers
// compile commands for files not present in the database.
class JSONBuildDatabasePlugin : public CompilationDatabasePlugin {
  std::unique_ptr<CompilationDatabase>
  loadFromDirectory(StringRef Directory, std::string &ErrorMessage) override {
    SmallString<1024> JSONDatabasePath(Directory);
    llvm::sys::path::append(JSONDatabasePath, "build_database.json");
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
  const auto TURefI = IndexByFile.find(Match);
  if (TURefI == IndexByFile.end())
    return {};
  std::vector<CompileCommand> Commands;
  getCommands(TURefI->getValue(), Commands);
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

const ModuleManager *JSONBuildDatabase::getModuleManager() const {
  return this;
}

std::vector<std::string>
JSONBuildDatabase::getRequiredModules(StringRef FilePath) const {
  const auto *TURef = getTUForSource(FilePath);
  std::vector<std::string> RequiredModules;
  if (TURef) {
    for (const auto &RequiredModule : TURef->RequiredModules) {
      SmallString<8> RequiredModuleStorage;
      RequiredModules.emplace_back(
          RequiredModule->getValue(RequiredModuleStorage));
    }
  }
  return RequiredModules;
}

std::optional<std::string>
JSONBuildDatabase::getModuleName(StringRef FilePath) const {
  const auto *TURef = getTUForSource(FilePath);
  if (TURef && TURef->ProvidesModuleName) {
    SmallString<8> ModuleNameStorage;
    return TURef->ProvidesModuleName->getValue(ModuleNameStorage).str();
  }
  return std::nullopt;
}

ModuleManager::ModuleNameState
JSONBuildDatabase::getModuleNameState(StringRef ModuleName) const {
  auto It = ModuleNameToDistinctSources.find(ModuleName);
  if (It == ModuleNameToDistinctSources.end())
    return ModuleNameState::Unknown;
  return It->second.size() > 1 ? ModuleNameState::Multiple
                               : ModuleNameState::Unique;
}

std::string
JSONBuildDatabase::getSourceForModuleName(StringRef ModuleName,
                                          StringRef RequiredSourceFile) const {
  const auto *RequiredSourceTURef = getTUForSource(RequiredSourceFile);
  if (RequiredSourceTURef) {
    SmallString<8> SetNameStorage;
    SmallString<8> FilenameStorage;
    auto SetName = RequiredSourceTURef->SetName->getValue(SetNameStorage);
    // First attempt to find the matching module in the current set
    const auto *ModuleTURef = getTUForModule(ModuleName, SetName);
    if (ModuleTURef)
      return ModuleTURef->Filename->getValue(FilenameStorage).str();
    // Could not find in current set, check all visible sets
    const auto TUSetI = IndexBySet.find(SetName);
    if (TUSetI == IndexBySet.end())
      return {};
    for (const auto &VisibleSet : TUSetI->getValue().VisibleSets) {
      ModuleTURef =
          getTUForModule(ModuleName, VisibleSet->getValue(SetNameStorage));
      if (ModuleTURef)
        return ModuleTURef->Filename->getValue(FilenameStorage).str();
    }
  }
  return {};
}

const JSONBuildDatabase::TranslationUnitRef *
JSONBuildDatabase::getTUForSource(StringRef FilePath) const {
  SmallString<128> NativeFilePath;
  llvm::sys::path::native(FilePath, NativeFilePath);

  std::string Error;
  llvm::raw_string_ostream ES(Error);
  StringRef Match = MatchTrie.findEquivalent(NativeFilePath, ES);
  if (Match.empty())
    return {};
  const auto TURefI = IndexByFile.find(Match);
  if (TURefI == IndexByFile.end())
    return {};
  // Return the first reference in the build database
  // Not ideal, but without context this is the best we can do
  for (const auto &TURef : TURefI->getValue()) {
    return &TURef;
  }
  return nullptr;
}

const JSONBuildDatabase::TranslationUnitRef *
JSONBuildDatabase::getTUForModule(StringRef ModuleName,
                                  StringRef SetName) const {
  const auto TUSetI = IndexBySet.find(SetName);
  if (TUSetI == IndexBySet.end())
    return {};
  // Return the first reference in the build database
  // Not ideal, but without context this is the best we can do
  for (const auto &TURef : TUSetI->getValue().TranslationUnits) {

    SmallString<8> ModuleNameStorage;
    if (TURef.ProvidesModuleName &&
        TURef.ProvidesModuleName->getValue(ModuleNameStorage) == ModuleName)
      return &TURef;
  }
  return nullptr;
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
    // The wrappers themselves don't take flags, so Args[1] is a compiler
    // flag, an input file, or a compiler. Inputs have extensions, compilers
    // don't.
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
  // There may be multiple wrappers: using distcc and ccache together is
  // common.
  while (unwrapCommand(Arguments))
    ;
  return Arguments;
}

void JSONBuildDatabase::getCommands(
    ArrayRef<TranslationUnitRef> TUsRef,
    std::vector<CompileCommand> &Commands) const {
  for (const auto &TURef : TUsRef) {
    SmallString<8> DirectoryStorage;
    SmallString<32> FilenameStorage;
    SmallString<32> OutputStorage;
    Commands.emplace_back(TURef.Directory->getValue(DirectoryStorage),
                          TURef.Filename->getValue(FilenameStorage),
                          nodeToCommandLine(TURef.CommandLine),
                          TURef.Output ? TURef.Output->getValue(OutputStorage)
                                       : "");
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
  SmallString<10> VersionStorage;
  if (Version->getValue(VersionStorage).str() != "1") {
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
  std::vector<llvm::yaml::ScalarNode *> VisibleSetsRefs = {};
  llvm::yaml::SequenceNode *TUs = nullptr;
  std::vector<TranslationUnitRef> TURefs = {};
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
      for (auto &VisibleSet : *VisibleSets) {
        auto *Scalar = dyn_cast<llvm::yaml::ScalarNode>(&VisibleSet);
        if (!Scalar) {
          ErrorMessage = "Only strings are allowed in 'visible-sets'.";
          return false;
        }
        VisibleSetsRefs.push_back(Scalar);
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
        TranslationUnitRef TURef = {};
        if (!parseTU(ErrorMessage, TUObject, TURef)) {
          return false;
        }
        TURefs.push_back(std::move(TURef));
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
  // Finalize the translation unit refs now that we have all the set info
  for (auto &TURef : TURefs) {
    // Attach the parent set name for easy lookups
    TURef.SetName = Name;
    // Build up the native file path
    SmallString<8> FileStorage;
    StringRef FileName = TURef.Filename->getValue(FileStorage);
    SmallString<128> NativeFilePath;
    if (llvm::sys::path::is_relative(FileName)) {
      SmallString<8> DirectoryStorage;
      SmallString<128> AbsolutePath(
          TURef.Directory->getValue(DirectoryStorage));
      llvm::sys::path::append(AbsolutePath, FileName);
      llvm::sys::path::native(AbsolutePath, NativeFilePath);
    } else {
      llvm::sys::path::native(FileName, NativeFilePath);
    }
    llvm::sys::path::remove_dots(NativeFilePath, /*remove_dot_dot=*/true);
    IndexByFile[NativeFilePath].push_back(TURef);
    AllCommands.push_back(TURef);
    MatchTrie.insert(NativeFilePath);
    if (TURef.ProvidesModuleName) {
      SmallString<8> ModuleNameStorage;
      ModuleNameToDistinctSources
          [TURef.ProvidesModuleName->getValue(ModuleNameStorage).str()]
              .insert(NativeFilePath);
    }
  }
  // Generate lookup for each set
  TranslationUnitSet TUSet = {};
  TUSet.TranslationUnits = std::move(TURefs);
  TUSet.VisibleSets = std::move(VisibleSetsRefs);
  SmallString<8> NameStorage;
  IndexBySet[Name->getValue(NameStorage)] = std::move(TUSet);
  return true;
}

bool JSONBuildDatabase::parseTU(std::string &ErrorMessage,
                                llvm::yaml::MappingNode *TUObject,
                                TranslationUnitRef &TURef) {
  llvm::yaml::SequenceNode *Arguments = nullptr;
  std::vector<llvm::yaml::ScalarNode *> Command;
  llvm::yaml::ScalarNode *Language = nullptr;
  llvm::yaml::SequenceNode *LocalArguments = nullptr;
  llvm::yaml::ScalarNode *WorkDirectory = nullptr;
  llvm::yaml::ScalarNode *Private = nullptr;
  llvm::yaml::ScalarNode *Source = nullptr;
  llvm::yaml::ScalarNode *Object = nullptr;
  llvm::yaml::MappingNode *Provides = nullptr;
  llvm::yaml::ScalarNode *ProvidesModuleName = nullptr;
  llvm::yaml::ScalarNode *ProvidesModulePCM = nullptr;
  llvm::yaml::SequenceNode *Requires = nullptr;
  std::vector<llvm::yaml::ScalarNode *> RequiredModules;
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
      for (auto &NextProvidesKeyValue : *Provides) {
        // The spec allows multiple of module provide, but C++ only allows one
        if (ProvidesModuleName) {
          ErrorMessage = "TU can only provide one module.";
          return false;
        }

        auto *ProvidesKeyString = dyn_cast_if_present<llvm::yaml::ScalarNode>(
            NextProvidesKeyValue.getKey());
        if (!ProvidesKeyString) {
          ErrorMessage = "Expected strings as key.";
          return false;
        }
        auto *ProvidesValue = dyn_cast_if_present<llvm::yaml::ScalarNode>(
            NextProvidesKeyValue.getValue());
        if (!ProvidesValue) {
          ErrorMessage = "Expected string as provides value.";
          return false;
        }

        ProvidesModuleName = ProvidesKeyString;
        ProvidesModulePCM = ProvidesValue;
      }
    } else if (KeyValue == "requires") {
      Requires = dyn_cast<llvm::yaml::SequenceNode>(Value);
      if (!Requires) {
        ErrorMessage = "Expected array as value for \"requires\".";
        return false;
      }
      for (auto &RequiredModule : *Requires) {
        auto *Scalar = dyn_cast<llvm::yaml::ScalarNode>(&RequiredModule);
        if (!Scalar) {
          ErrorMessage = "Only strings are allowed in 'requires'.";
          return false;
        }
        RequiredModules.push_back(Scalar);
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
  TURef.Directory = WorkDirectory;
  TURef.Filename = Source;
  TURef.CommandLine = std::move(Command);
  TURef.Output = Object;
  TURef.ProvidesModuleName = ProvidesModuleName;
  TURef.ProvidesModulePCM = ProvidesModulePCM;
  TURef.RequiredModules = std::move(RequiredModules);
  return true;
}
