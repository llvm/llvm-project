//===- JSONBuildDatabase.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  The JSONBuildDatabase finds build databases supplied as a file
//  'build_database.json'.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_JSONBUILDDATABASE_H
#define LLVM_CLANG_TOOLING_JSONBUILDDATABASE_H

#include "clang/Basic/LLVM.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/FileMatchTrie.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace clang {
namespace tooling {

/// A JSON based build database.
///
/// JSON build database files must contain a collection of JSON objects which
/// provide the sets of translation units which capture the visiblity for producing
/// and consuming dependency modules along with the compilation commands to build each TU.
/// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2977r2.html
/// {
///   "version": 1,
///   "revision": 0,
///   "sets": [
///     {
///       "family-name" : "<optional shared family name to semantically group sets>",
///       "name" : "<unique name for each set>",
///       "translation-units" : [
///         {
///           "arguments": [
///             "/path/to/compiler",
///             "...",
///           ],
///           "baseline-arguments" :
///           [
///             "...",
///           ],
///           "local-arguments" :
///           [
///             "...",
///           ],
///           "object": "<optional object file produced>",
///           "private": false,
///           "provides": {
///             "<module name>": "<path to BMI>"
///           },
///           "requires" : ["<list of module names that are imported>"],
///           "source": "<source file>",
///           "work-directory": "<working directory of the compile>"
///         },
///         ...
///       ],
///       "visible-sets" : ["<list of sets that are visible from this set>"]
///     }
///   ]
/// }
///
/// JSON build databases can for example be generated in CMake projects
/// by setting the flag -DCMAKE_EXPORT_BUILD_DATABASE.
class JSONBuildDatabase : public CompilationDatabase, ModuleManager {
public:
  /// Loads a JSON build database from the specified file.
  ///
  /// Returns NULL and sets ErrorMessage if the database could not be
  /// loaded from the given file.
  static std::unique_ptr<JSONBuildDatabase>
  loadFromFile(StringRef FilePath, std::string &ErrorMessage);

  /// Loads a JSON build database from a data buffer.
  ///
  /// Returns NULL and sets ErrorMessage if the database could not be loaded.
  static std::unique_ptr<JSONBuildDatabase>
  loadFromBuffer(StringRef DatabaseString, std::string &ErrorMessage);

  /// Returns all compile commands in which the specified file was
  /// compiled.
  ///
  /// FIXME: Currently FilePath must be an absolute path inside the
  /// source directory which does not have symlinks resolved.
  std::vector<CompileCommand>
  getCompileCommands(StringRef FilePath) const override;

  /// Returns the list of all files available in the build database.
  ///
  /// These are the 'file' entries of the JSON objects.
  std::vector<std::string> getAllFiles() const override;

  /// Returns all compile commands for all the files in the build
  /// database.
  std::vector<CompileCommand> getAllCompileCommands() const override;

  const ModuleManager *getModuleManager() const override;

  std::vector<std::string>
  getRequiredModules(StringRef FilePath) const override;
  std::optional<std::string> getModuleName(StringRef FilePath) const override;

  ModuleNameState getModuleNameState(StringRef ModuleName) const override;

  std::string
  getSourceForModuleName(StringRef ModuleName,
                         StringRef RequiredSourceFile) const override;

private:
  /// Constructs a JSON build database on a memory buffer.
  JSONBuildDatabase(std::unique_ptr<llvm::MemoryBuffer> Database)
      : Database(std::move(Database)),
        YAMLStream(this->Database->getBuffer(), SM) {}

  // Container for a compile command references where 'commandline'
  // points to the corresponding scalar nodes in the YAML stream.
  // The output field may be a nullptr.
  struct TranslationUnitRef {
    llvm::yaml::ScalarNode *SetName;
    llvm::yaml::ScalarNode *Directory;
    llvm::yaml::ScalarNode *Filename;
    std::vector<llvm::yaml::ScalarNode *> CommandLine;
    llvm::yaml::ScalarNode *Output;
    llvm::yaml::ScalarNode *ProvidesModuleName;
    llvm::yaml::ScalarNode *ProvidesModulePCM;
    std::vector<llvm::yaml::ScalarNode *> RequiredModules;
  };

  struct TranslationUnitSet {
    std::vector<llvm::yaml::ScalarNode *> VisibleSets;
    std::vector<TranslationUnitRef> TranslationUnits;
  };

  /// Parses the database file and creates the index.
  ///
  /// Returns whether parsing succeeded. Sets ErrorMessage if parsing
  /// failed.
  bool parse(std::string &ErrorMessage);
  bool parseRoot(std::string &ErrorMessage, llvm::yaml::MappingNode *Object);
  bool parseSet(std::string &ErrorMessage, llvm::yaml::MappingNode *Object);
  bool parseTU(std::string &ErrorMessage, llvm::yaml::MappingNode *Object,
               TranslationUnitRef &TURef);

  const TranslationUnitRef *getTUForSource(StringRef FilePath) const;
  const TranslationUnitRef *getTUForModule(StringRef ModuleName,
                                           StringRef SetName) const;

  /// Converts the given array of TranslationUnitRefs to CompileCommands.
  void getCommands(ArrayRef<TranslationUnitRef> CommandsRef,
                   std::vector<CompileCommand> &Commands) const;

  // Maps file paths to the translation units for that file.
  llvm::StringMap<std::vector<TranslationUnitRef>> IndexByFile;
  llvm::StringMap<TranslationUnitSet> IndexBySet;

  /// All the compile commands in the order that they were provided in the
  /// JSON stream.
  std::vector<TranslationUnitRef> AllCommands;

  // Module name state lookup to track unique names
  using DistinctSourceSet = llvm::StringSet<>;
  llvm::StringMap<DistinctSourceSet> ModuleNameToDistinctSources;

  FileMatchTrie MatchTrie;

  std::unique_ptr<llvm::MemoryBuffer> Database;
  llvm::SourceMgr SM;
  llvm::yaml::Stream YAMLStream;
};

} // namespace tooling
} // namespace clang

#endif // LLVM_CLANG_TOOLING_JSONBUILDDATABASE_H
