//===--- DeviceCodeObjectLinker.cpp - Device Code Object Linker -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceCodeObjectLinker.h"

#include "clang/Options/Options.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"

#include <memory>

namespace clang {

llvm::Expected<std::unique_ptr<DeviceCodeObjectLinker>>
selectDeviceCodeObjectLinker(llvm::ArrayRef<const char *> DriverArgs) {
  unsigned MissingArgIndex, MissingArgCount;
  llvm::opt::InputArgList Args = getDriverOptTable().ParseArgs(
      DriverArgs, MissingArgIndex, MissingArgCount,
      llvm::opt::Visibility(options::ClangOption));

  // An explicit --ld-path means the user asked for an external linker.
  // FIXME: Return the external linker asked.
  if (Args.hasArg(options::OPT_ld_path_EQ))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "No device code object linker is available.");

  // FIXME: Return an in-process (JITLink-based) linker.
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "No device code object linker is available.");
}

} // namespace clang
