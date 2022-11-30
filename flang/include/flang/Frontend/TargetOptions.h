//===--- TargetOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the flang::TargetOptions class.
///
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_FRONTEND_TARGETOPTIONS_H
#define FORTRAN_FRONTEND_TARGETOPTIONS_H

#include <string>

namespace Fortran::frontend {

/// Options for controlling the target.
class TargetOptions {
public:
  /// The name of the target triple to compile for.
  std::string triple;

  /// If given, the name of the target CPU to generate code for.
  std::string cpu;

  /// The list of target specific features to enable or disable, as written on
  /// the command line.
  std::vector<std::string> featuresAsWritten;
};

} // end namespace Fortran::frontend

#endif
