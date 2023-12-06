//===--- DebugOptions.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DebugOptions interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_DEBUGOPTIONS_H
#define LLVM_CLANG_BASIC_DEBUGOPTIONS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/Debug/Options.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/Compression.h"
#include "llvm/Target/TargetOptions.h"
#include <string>

namespace clang {

/// Bitfields of DebugOptions, split out from DebugOptions to ensure
/// that this large collection of bitfields is a trivial class type.
class DebugOptionsBase {
  friend class CompilerInvocation;
  friend class CompilerInvocationBase;

public:
#define DEBUGOPT(Name, Bits, Default) unsigned Name : Bits;
#define ENUM_DEBUGOPT(Name, Type, Bits, Default)
#include "clang/Basic/DebugOptions.def"

protected:
#define DEBUGOPT(Name, Bits, Default)
#define ENUM_DEBUGOPT(Name, Type, Bits, Default) unsigned Name : Bits;
#include "clang/Basic/DebugOptions.def"
};

/// DebugOptions - Track various options which control how the debug information
/// is generated for the backend.
class DebugOptions : public DebugOptionsBase {
public:
  enum DebugSrcHashKind {
    DSH_MD5,
    DSH_SHA1,
    DSH_SHA256,
  };

  /// Enable additional debugging information.
  std::string DebugPass;

  /// The string to embed in debug information as the current working directory.
  std::string DebugCompilationDir;

  /// The string to embed in the debug information for the compile unit, if
  /// non-empty.
  std::string DwarfDebugFlags;

  enum AssignmentTrackingOpts {
    Disabled,
    Enabled,
    Forced,
  };

  llvm::SmallVector<std::pair<std::string, std::string>, 0> DebugPrefixMap;

  /// The file to use for dumping bug report by `Debugify` for original
  /// debug info.
  std::string DIBugsReportFilePath;

  /// The name for the split debug info file used for the DW_AT_[GNU_]dwo_name
  /// attribute in the skeleton CU.
  std::string SplitDwarfFile;

  /// Output filename for the split debug info, not used in the skeleton CU.
  std::string SplitDwarfOutput;

  /// Output filename used in the COFF debug information.
  std::string ObjectFilenameForDebug;

public:
  // Define accessors/mutators for code generation options of enumeration type.
#define DEBUGOPT(Name, Bits, Default)
#define ENUM_DEBUGOPT(Name, Type, Bits, Default)                               \
  Type get##Name() const { return static_cast<Type>(Name); }                   \
  void set##Name(Type Value) { Name = static_cast<unsigned>(Value); }
#include "clang/Basic/DebugOptions.def"

  DebugOptions();

  /// Check if type and variable info should be emitted.
  bool hasReducedDebugInfo() const {
    return getDebugInfo() >= llvm::debugoptions::DebugInfoConstructor;
  }

  /// Check if maybe unused type info should be emitted.
  bool hasMaybeUnusedDebugInfo() const {
    return getDebugInfo() >= llvm::debugoptions::UnusedTypeInfo;
  }

  /// Reset all of the options that are not considered when building a
  /// module.
  void resetNonModularOptions(llvm::StringRef ModuleFormat);
};

} // end namespace clang

#endif
