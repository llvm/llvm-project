//===- OffloadBundler.h - File Bundling and Unbundling ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines an offload bundling API that bundles different files
/// that relate with the same source code but different targets into a single
/// one. Also the implements the opposite functionality, i.e. unbundle files
/// previous created by this API.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_OFFLOADBUNDLER_H
#define LLVM_CLANG_DRIVER_OFFLOADBUNDLER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorWithFlags.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Triple.h"
#include <llvm/Support/MemoryBuffer.h>
#include <string>

namespace clang {

class OffloadBundlerConfig {
public:
  OffloadBundlerConfig();

  bool getAllowNoHost() const { return OutputFileNames.getFlag(0); }
  void setAllowNoHost(bool Value) { OutputFileNames.setFlag(0, Value); }
  bool getAllowMissingBundles() const { return OutputFileNames.getFlag(1); }
  void setAllowMissingBundles(bool Value) { OutputFileNames.setFlag(1, Value); }
  bool getCheckInputArchive() const { return OutputFileNames.getFlag(2); }
  void setCheckInputArchive(bool Value) { OutputFileNames.setFlag(2, Value); }
  bool getPrintExternalCommands() const { return OutputFileNames.getFlag(3); }
  void setPrintExternalCommands(bool Value) {
    OutputFileNames.setFlag(3, Value);
  }
  bool getHipOpenmpCompatible() const { return InputFileNames.getFlag(0); }
  void setHipOpenmpCompatible(bool Value) { InputFileNames.setFlag(0, Value); }
  bool getCompress() const { return InputFileNames.getFlag(1); }
  bool setCompress(bool Value) {
    InputFileNames.setFlag(1, Value);
    return Value;
  }
  bool getVerbose() const { return InputFileNames.getFlag(2); }
  bool setVerbose(bool Value) {
    InputFileNames.setFlag(2, Value);
    return Value;
  }
  llvm::compression::Format CompressionFormat;
  int CompressionLevel;
  uint16_t CompressedBundleVersion;

  unsigned BundleAlignment = 1;
  unsigned HostInputIndex = ~0u;

  std::string FilesType;
  std::string ObjcopyPath;

  llvm::SmallVector<std::string, 4> TargetNames;
  llvm::SmallVectorWithFlags<std::string, 4, 3> InputFileNames;
  llvm::SmallVectorWithFlags<std::string, 4, 4> OutputFileNames;
};

class OffloadBundler {
public:
  const OffloadBundlerConfig &BundlerConfig;

  // TODO: Add error checking from ClangOffloadBundler.cpp
  OffloadBundler(const OffloadBundlerConfig &BC) : BundlerConfig(BC) {}

  // List bundle IDs. Return true if an error was found.
  static llvm::Error
  ListBundleIDsInFile(llvm::StringRef InputFileName,
                      const OffloadBundlerConfig &BundlerConfig);

  llvm::Error BundleFiles();
  llvm::Error UnbundleFiles();
  llvm::Error UnbundleArchive();
};

/// Obtain the offload kind, real machine triple, and an optional TargetID
/// out of the target information specified by the user.
/// Bundle Entry ID (or, Offload Target String) has following components:
///  * Offload Kind - Host, OpenMP, or HIP
///  * Triple - Standard LLVM Triple
///  * TargetID (Optional) - target ID, like gfx906:xnack+ or sm_30
struct OffloadTargetInfo {
  llvm::StringRef OffloadKind;
  llvm::Triple Triple;
  llvm::StringRef TargetID;

  const OffloadBundlerConfig &BundlerConfig;

  OffloadTargetInfo(const llvm::StringRef Target,
                    const OffloadBundlerConfig &BC);
  bool hasHostKind() const;
  bool isOffloadKindValid() const;
  bool isOffloadKindCompatible(const llvm::StringRef TargetOffloadKind) const;
  bool isTripleValid() const;
  bool operator==(const OffloadTargetInfo &Target) const;
  std::string str() const;
};

/// Check whether the bundle id is in the following format:
/// <kind>-<triple>[-<target id>[:target features]]
/// <triple> := <arch>-<vendor>-<os>-<env>
bool checkOffloadBundleID(const llvm::StringRef Str);
} // namespace clang

#endif // LLVM_CLANG_DRIVER_OFFLOADBUNDLER_H
