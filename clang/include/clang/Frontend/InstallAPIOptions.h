//===--- InstallAPIOptions.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_INSTALLAPIOPTIONS_H
#define LLVM_CLANG_FRONTEND_INSTALLAPIOPTIONS_H

#include "llvm/TextAPI/PackedVersion.h"

namespace clang {

/// InstallAPIOptions - Options for controlling InstallAPI verification and
/// TextAPI output.
class InstallAPIOptions {
public:
  /// The install name which is apart of the library's ID.
  std::string InstallName;

  /// The current version which is apart of the library's ID.
  llvm::MachO::PackedVersion CurrentVersion;
};
} // namespace clang

#endif
