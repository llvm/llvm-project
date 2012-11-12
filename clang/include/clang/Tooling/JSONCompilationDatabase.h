//===--- JSONCompilationDatabase.h - ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  The JSONCompilationDatabase finds compilation databases supplied as a file
//  'compile_commands.json'.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_JSON_COMPILATION_DATABASE_H
#define LLVM_CLANG_TOOLING_JSON_COMPILATION_DATABASE_H

#include "clang/Basic/LLVM.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/FileMatchTrie.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include <string>
#include <vector>

namespace clang {
namespace tooling {

/// \brief A JSON based compilation database.
///
/// JSON compilation database files must contain a list of JSON objects which
/// provide the command lines in the attributes 'directory', 'command' and
/// 'file':
/// [
///   { "directory": "<working directory of the compile>",
///     "command": "<compile command line>",
///     "file": "<path to source file>"
///   },
///   ...
/// ]
/// Each object entry defines one compile action. The specified file is
/// considered to be the main source file for the translation unit.
///
/// JSON compilation databases can for example be generated in CMake projects
/// by setting the flag -DCMAKE_EXPORT_COMPILE_COMMANDS.
class JSONCompilationDatabase : public CompilationDatabase {
public:
  /// \brief Loads a JSON compilation database from the specified file.
  ///
  /// Returns NULL and sets ErrorMessage if the database could not be
  /// loaded from the given file.
  static JSONCompilationDatabase *loadFromFile(StringRef FilePath,
                                               std::string &ErrorMessage);

  /// \brief Loads a JSON compilation database from a data buffer.
  ///
  /// Returns NULL and sets ErrorMessage if the database could not be loaded.
  static JSONCompilationDatabase *loadFromBuffer(StringRef DatabaseString,
                                                 std::string &ErrorMessage);

  /// \brief Returns all compile comamnds in which the specified file was
  /// compiled.
  ///
  /// FIXME: Currently FilePath must be an absolute path inside the
  /// source directory which does not have symlinks resolved.
  virtual std::vector<CompileCommand> getCompileCommands(
    StringRef FilePath) const;

  /// \brief Returns the list of all files available in the compilation database.
  ///
  /// These are the 'file' entries of the JSON objects.
  virtual std::vector<std::string> getAllFiles() const;

private:
  /// \brief Constructs a JSON compilation database on a memory buffer.
  JSONCompilationDatabase(llvm::MemoryBuffer *Database)
    : Database(Database), YAMLStream(Database->getBuffer(), SM) {}

  /// \brief Parses the database file and creates the index.
  ///
  /// Returns whether parsing succeeded. Sets ErrorMessage if parsing
  /// failed.
  bool parse(std::string &ErrorMessage);

  // Tuple (directory, commandline) where 'commandline' pointing to the
  // corresponding nodes in the YAML stream.
  typedef std::pair<llvm::yaml::ScalarNode*,
                    llvm::yaml::ScalarNode*> CompileCommandRef;

  // Maps file paths to the compile command lines for that file.
  llvm::StringMap< std::vector<CompileCommandRef> > IndexByFile;

  FileMatchTrie MatchTrie;

  llvm::OwningPtr<llvm::MemoryBuffer> Database;
  llvm::SourceMgr SM;
  llvm::yaml::Stream YAMLStream;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_JSON_COMPILATION_DATABASE_H
