//===--- Tool.cpp - Compilation Tools -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Tool.h"
#include "InputInfo.h"

using namespace clang::driver;

Tool::Tool(const char *_Name, const char *_ShortName, const ToolChain &TC,
           ResponseFileSupport _ResponseSupport,
           llvm::sys::WindowsEncodingMethod _ResponseEncoding,
           const char *_ResponseFlag)
    : Name(_Name), ShortName(_ShortName), TheToolChain(TC),
      ResponseSupport(_ResponseSupport), ResponseEncoding(_ResponseEncoding),
      ResponseFlag(_ResponseFlag) {}

Tool::~Tool() {
}

void Tool::ConstructJobMultipleOutputs(Compilation &C, const JobAction &JA,
                                       const InputInfoList &Outputs,
                                       const InputInfoList &Inputs,
                                       const llvm::opt::ArgList &TCArgs,
                                       const char *LinkingOutput) const {
  assert(Outputs.size() == 1 && "Expected only one output by default!");
  ConstructJob(C, JA, Outputs.front(), Inputs, TCArgs, LinkingOutput);
}
