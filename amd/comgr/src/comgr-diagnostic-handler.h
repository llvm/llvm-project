//===- comgr-diagnostic-handler.h - Handle LLVM diagnostics ---------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_DIAGNOSTIC_HANDLER_H
#define COMGR_DIAGNOSTIC_HANDLER_H

#include <llvm/IR/DiagnosticInfo.h>

namespace COMGR {
struct AMDGPUCompilerDiagnosticHandler : public llvm::DiagnosticHandler {
  llvm::raw_ostream &LogS;

  AMDGPUCompilerDiagnosticHandler(llvm::raw_ostream &LogS) : LogS(LogS) {}

  bool handleDiagnostics(const llvm::DiagnosticInfo &DI) override;
};
} // namespace COMGR

#endif
