//===--- TargetOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the clang::TargetOptions class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETOPTIONS_H
#define LLVM_CLANG_BASIC_TARGETOPTIONS_H

#include "clang/Basic/OpenCLOptions.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Target/TargetOptions.h"
#include <string>
#include <vector>

namespace clang {

/// Options for controlling the target.
class TargetOptions {
public:
  /// The name of the target triple to compile for.
  std::string Triple;

  /// When compiling for the device side, contains the triple used to compile
  /// for the host.
  std::string HostTriple;

  /// If given, the name of the target CPU to generate code for.
  std::string CPU;

  /// If given, the name of the target CPU to tune code for.
  std::string TuneCPU;

  /// If given, the unit to use for floating point math.
  std::string FPMath;

  /// If given, the name of the target ABI to use.
  std::string ABI;

  /// The EABI version to use
  llvm::EABI EABIVersion;

  /// If given, the version string of the linker in use.
  std::string LinkerVersion;

  /// The list of target specific features to enable or disable, as written on the command line.
  std::vector<std::string> FeaturesAsWritten;

  /// The list of target specific features to enable or disable -- this should
  /// be a list of strings starting with by '+' or '-'.
  std::vector<std::string> Features;

  /// The map of which features have been enabled disabled based on the command
  /// line.
  llvm::StringMap<bool> FeatureMap;

  /// Supported OpenCL extensions and optional core features.
  OpenCLOptions SupportedOpenCLOptions;

  /// The list of OpenCL extensions to enable or disable, as written on
  /// the command line.
  std::vector<std::string> OpenCLExtensionsAsWritten;

  /// If given, enables support for __int128_t and __uint128_t types.
  bool ForceEnableInt128 = false;

  /// \brief If enabled, use 32-bit pointers for accessing const/local/shared
  /// address space.
  bool NVPTXUseShortPointers = false;

  // The code model to be used as specified by the user. Corresponds to
  // CodeModel::Model enum defined in include/llvm/Support/CodeGen.h, plus
  // "default" for the case when the user has not explicitly specified a
  // code model.
  std::string CodeModel;

  /// The version of the SDK which was used during the compilation.
  /// The option is used for two different purposes:
  /// * on darwin the version is propagated to LLVM where it's used
  ///   to support SDK Version metadata (See D55673).
  /// * CUDA compilation uses it to control parts of CUDA compilation
  ///   in clang that depend on specific version of the CUDA SDK.
  llvm::VersionTuple SDKVersion;
};

}  // end namespace clang

#endif
