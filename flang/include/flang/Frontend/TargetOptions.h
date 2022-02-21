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

#ifndef LLVM_FLANG_TARGETOPTIONS_H
#define LLVM_FLANG_TARGETOPTIONS_H

namespace Fortran::frontend {

/// Options for controlling the target. Currently this is just a placeholder.
/// In the future, we will use this to specify various target options that
/// will affect the generated code e.g.:
///   * CPU to tune the code for
///   * available CPU/hardware extensions
///   * target specific features to enable/disable
///   * options for accelerators (e.g. GPUs)
///   * (...)
class TargetOptions {
public:
  /// The name of the target triple to compile for.
  std::string triple;
};

} // end namespace Fortran::frontend

#endif
