//===-- APINotesYAMLCompiler.h - API Notes YAML Format Reader ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_APINOTES_APINOTESYAMLCOMPILER_H
#define LLVM_CLANG_APINOTES_APINOTESYAMLCOMPILER_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
class DarwinSDKInfo;
class FileEntry;
} // namespace clang

namespace clang {
namespace api_notes {
/// Parses the APINotes YAML content and writes the representation back to the
/// specified stream.  This provides a means of testing the YAML processing of
/// the APINotes format.
bool parseAndDumpAPINotes(llvm::StringRef YI, llvm::raw_ostream &OS);

/// Resolves the SDK being compiled against, on demand. Returns null when the
/// SDK cannot be identified (no sysroot, no SDKSettings.json, ...), which means
/// "apply the API notes" rather than "skip them".
using DarwinSDKInfoProviderRef = llvm::function_ref<const DarwinSDKInfo *()>;

enum class CompileResult {
  Success,
  Error,
  /// The file declares 'ValidSDKs' and the SDK being compiled against isn't one
  /// of them, so nothing was written.
  Skipped,
};

/// Converts API notes from YAML format to binary format.
CompileResult compileAPINotes(
    llvm::StringRef YAMLInput, const FileEntry *SourceFile,
    llvm::raw_ostream &OS, llvm::SourceMgr::DiagHandlerTy DiagHandler = nullptr,
    void *DiagHandlerCtxt = nullptr, DarwinSDKInfoProviderRef GetSDKInfo = {});
} // namespace api_notes
} // namespace clang

#endif
