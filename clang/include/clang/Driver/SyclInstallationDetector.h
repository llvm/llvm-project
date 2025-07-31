//===-- SyclInstallationDetector.h - SYCL Instalation Detector --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_SYCLINSTALLATIONDETECTOR_H
#define LLVM_CLANG_DRIVER_SYCLINSTALLATIONDETECTOR_H

#include "clang/Driver/Driver.h"

namespace clang {
namespace driver {

class SYCLInstallationDetector {
public:
  SYCLInstallationDetector(const Driver &D, const llvm::Triple &HostTriple,
                           const llvm::opt::ArgList &Args);

  void addSYCLIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const;
};

} // namespace driver
} // namespace clang

#endif // LLVM_CLANG_DRIVER_SYCLINSTALLATIONDETECTOR_H
