//===--- TargetOptions.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::TargetOptions class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_TARGETOPTIONS_H
#define LLVM_CLANG_FRONTEND_TARGETOPTIONS_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include <string>
#include <vector>

namespace clang {

/// \brief Options for controlling the target.
class TargetOptions : public RefCountedBase<TargetOptions> {
public:
  /// If given, the name of the target triple to compile for. If not given the
  /// target will be selected to match the host.
  std::string Triple;

  /// If given, the name of the target CPU to generate code for.
  std::string CPU;

  /// If given, the name of the target ABI to use.
  std::string ABI;

  /// If given, the name of the target C++ ABI to use. If not given, defaults
  /// to "itanium".
  std::string CXXABI;

  /// If given, the version string of the linker in use.
  std::string LinkerVersion;

  /// \brief The list of target specific features to enable or disable, as written on the command line.
  std::vector<std::string> FeaturesAsWritten;

  /// The list of target specific features to enable or disable -- this should
  /// be a list of strings starting with by '+' or '-'.
  std::vector<std::string> Features;
};

}  // end namespace clang

#endif
