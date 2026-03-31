//===- Utils.h - Shared utilities for SSAF tools ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Shared error-handling, format-registry cache, and summary-file abstraction
//  used by clang-ssaf-linker and clang-ssaf-format.
//
//  All declarations live in the clang::ssaf namespace.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_TOOL_UTILS_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_TOOL_UTILS_H

#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/SerializationFormatRegistry.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>

namespace clang::ssaf {

//===----------------------------------------------------------------------===//
// Tool Identity
//===----------------------------------------------------------------------===//

/// Returns the name of the running tool, as set by initTool().
llvm::StringRef getToolName();

//===----------------------------------------------------------------------===//
// Error Messages
//===----------------------------------------------------------------------===//

namespace ErrorMessages {

constexpr const char *CannotValidateSummary =
    "failed to validate summary '{0}': {1}";

constexpr const char *ExtensionNotSupplied = "Extension not supplied";

constexpr const char *NoFormatForExtension =
    "Format not registered for extension '{0}'";

constexpr const char *OutputDirectoryMissing =
    "Parent directory does not exist";

constexpr const char *FailedToLoadPlugin = "failed to load plugin '{0}': {1}";

} // namespace ErrorMessages

//===----------------------------------------------------------------------===//
// Diagnostic Utilities
//===----------------------------------------------------------------------===//

[[noreturn]] void fail(const char *Msg);

template <typename... Ts>
[[noreturn]] inline void fail(const char *Fmt, Ts &&...Args) {
  std::string Message = llvm::formatv(Fmt, std::forward<Ts>(Args)...);
  fail(Message.data());
}

[[noreturn]] void fail(llvm::Error Err);

//===----------------------------------------------------------------------===//
// Plugin Loading
//===----------------------------------------------------------------------===//

void loadPlugins(llvm::ArrayRef<std::string> Paths);

//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

/// Sets ToolName, ToolVersion, and the version printer, hides unrelated
/// command-line options, and parses arguments. Must be called after InitLLVM.
void initTool(int argc, const char **argv, llvm::StringRef Version,
              llvm::cl::OptionCategory &Category, llvm::StringRef ToolHeading);

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

struct SummaryFile {
  std::string Path;
  SerializationFormat *Format = nullptr;

  /// Constructs a SummaryFile by resolving the serialization format from the
  /// file extension. Calls fail() and exits if the extension is missing or
  /// unregistered.
  static SummaryFile fromPath(llvm::StringRef Path);
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_TOOL_UTILS_H
