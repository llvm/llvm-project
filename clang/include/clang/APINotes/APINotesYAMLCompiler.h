//=== APINotesYAMLCompiler.h - API Notes YAML to binary compiler *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file reads sidecar API notes specified in YAML format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_API_NOTES_YAML_COMPILER_H
#define LLVM_CLANG_API_NOTES_YAML_COMPILER_H
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>

namespace llvm {
  class raw_ostream;
  class MemoryBuffer;
}

namespace clang {

class FileEntry;

namespace api_notes {

  enum class ActionType {
    None,
    YAMLToBinary,
    BinaryToYAML,
    Dump,
  };

  /// Converts API notes from YAML format to binary format.
  bool compileAPINotes(llvm::StringRef yamlInput,
                       const FileEntry *sourceFile,
                       llvm::raw_ostream &os,
                       llvm::SourceMgr::DiagHandlerTy diagHandler = nullptr,
                       void *diagHandlerCtxt = nullptr);

  bool parseAndDumpAPINotes(llvm::StringRef yamlInput);
} // end namespace api_notes
} // end namespace clang

#endif // LLVM_CLANG_API_NOTES_YAML_COMPILER_H
