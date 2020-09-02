//===- PreprocessorOptions.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PREPROCESSOROPTIONS_H_
#define LLVM_CLANG_LEX_PREPROCESSOROPTIONS_H_

#include "clang/Basic/LLVM.h"
#include "clang/Lex/PreprocessorExcludedConditionalDirectiveSkipMapping.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

class MemoryBuffer;

} // namespace llvm

namespace clang {

/// Enumerate the kinds of standard library that
enum ObjCXXARCStandardLibraryKind {
  ARCXX_nolib,

  /// libc++
  ARCXX_libcxx,

  /// libstdc++
  ARCXX_libstdcxx
};

/// PreprocessorOptions - This class is used for passing the various options
/// used in preprocessor initialization to InitializePreprocessor().
class PreprocessorOptions {
public:
  using MacrosTy = std::vector<std::pair<std::string, bool>>;
  using PrecompiledPreambleBytesTy = std::pair<unsigned, bool>;
  using RemappedFilesTy = std::vector<std::pair<std::string, std::string>>;
  using RemappedFileBuffersTy =
      std::vector<std::pair<std::string, llvm::MemoryBuffer *>>;
  using MacroPrefixMapTy =
      std::map<std::string, std::string, std::greater<std::string>>;

  /// Records the set of modules
  class FailedModulesSet {
    llvm::StringSet<> Failed;

  public:
    bool hasAlreadyFailed(StringRef module) {
      return Failed.count(module) > 0;
    }

    void addFailed(StringRef module) {
      Failed.insert(module);
    }
  };

#define TYPED_PREPROCESSOROPT(Type, Name, Description) Type Name;
#include "clang/Lex/PreprocessorOptions.def"

  PreprocessorOptions()
      : UsePredefines(true), DetailedRecord(false), PCHWithHdrStop(false),
        PCHWithHdrStopCreate(false), DisablePCHValidation(false),
        AllowPCHWithCompilerErrors(false), DumpDeserializedPCHDecls(false),
        PrecompiledPreambleBytes(0, false), GeneratePreamble(false),
        WriteCommentListToPCH(true), SingleFileParseMode(false),
        LexEditorPlaceholders(true), RemappedFilesKeepOriginalName(true),
        RetainRemappedFileBuffers(false),
        RetainExcludedConditionalBlocks(false),
        ObjCXXARCStandardLibrary(ARCXX_nolib),
        ExcludedConditionalDirectiveSkipMappings(nullptr),
        SetUpStaticAnalyzer(false), DisablePragmaDebugCrash(false) {}

  void addMacroDef(StringRef Name) {
    Macros.emplace_back(std::string(Name), false);
  }
  void addMacroUndef(StringRef Name) {
    Macros.emplace_back(std::string(Name), true);
  }

  void addRemappedFile(StringRef From, StringRef To) {
    RemappedFiles.emplace_back(std::string(From), std::string(To));
  }

  void addRemappedFile(StringRef From, llvm::MemoryBuffer *To) {
    RemappedFileBuffers.emplace_back(std::string(From), To);
  }

  void clearRemappedFiles() {
    RemappedFiles.clear();
    RemappedFileBuffers.clear();
  }

  /// Reset any options that are not considered when building a
  /// module.
  void resetNonModularOptions() {
    Includes.clear();
    MacroIncludes.clear();
    ChainedIncludes.clear();
    DumpDeserializedPCHDecls = false;
    ImplicitPCHInclude.clear();
    SingleFileParseMode = false;
    LexEditorPlaceholders = true;
    RetainRemappedFileBuffers = true;
    PrecompiledPreambleBytes.first = 0;
    PrecompiledPreambleBytes.second = false;
    RetainExcludedConditionalBlocks = false;
  }
};

} // namespace clang

#endif // LLVM_CLANG_LEX_PREPROCESSOROPTIONS_H_
